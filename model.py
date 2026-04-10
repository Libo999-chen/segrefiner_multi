import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim=None, groups=32):
        super().__init__()
        gn_groups = groups if out_ch % groups == 0 else 8

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(gn_groups, out_ch)
        self.act = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(gn_groups, out_ch)

        self.shortcut = nn.Identity()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.GroupNorm(gn_groups, out_ch),
            )

        self.time_proj = None
        if time_dim is not None:
            self.time_proj = nn.Sequential(
                nn.Linear(time_dim, out_ch),
                nn.ReLU(inplace=True),
            )

    def forward(self, x, t_emb=None):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.gn1(out)

        if self.time_proj is not None and t_emb is not None:
            time_bias = self.time_proj(t_emb)[:, :, None, None]
            out = out + time_bias

        out = self.act(out)
        out = self.conv2(out)
        out = self.gn2(out)

        out = out + identity
        out = self.act(out)
        return out


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        attn_dim = max(in_dim // 8, 1)
        self.query = nn.Conv2d(in_dim, attn_dim, kernel_size=1)
        self.key = nn.Conv2d(in_dim, attn_dim, kernel_size=1)
        self.value = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.size()

        q = self.query(x).view(b, -1, h * w).permute(0, 2, 1)   # [B, HW, Cq]
        k = self.key(x).view(b, -1, h * w)                       # [B, Cq, HW]
        attn = torch.softmax(torch.bmm(q, k), dim=-1)           # [B, HW, HW]

        v = self.value(x).view(b, -1, h * w)                     # [B, C, HW]
        out = torch.bmm(v, attn.permute(0, 2, 1))               # [B, C, HW]
        out = out.view(b, c, h, w)

        return self.gamma * out + x


class TimeEmbedding(nn.Module):
    """
    Input:
        t: shape [B] or [B, 1], typically normalized to [0, 1]
    Output:
        shape [B, time_dim]
    """
    def __init__(self, time_dim=64):
        super().__init__()
        self.time_dim = time_dim
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.ReLU(inplace=True),
            nn.Linear(time_dim, time_dim),
        )

    def forward(self, t):
        if t.dim() == 2 and t.size(1) == 1:
            t = t.squeeze(1)
        elif t.dim() != 1:
            raise ValueError(f"Expected t to have shape [B] or [B,1], got {t.shape}")

        device = t.device
        half_dim = self.time_dim // 2
        freqs = torch.exp(
            -torch.arange(half_dim, device=device, dtype=torch.float32)
            * (torch.log(torch.tensor(10000.0, device=device)) / max(half_dim - 1, 1))
        )
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)

        if self.time_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return self.mlp(emb)


class UNet_Diffusion(nn.Module):
    """
    Multi-class refinement model for Cityscapes 6-class setup.

    Inputs:
        img  : [B, 3, H, W]
        mask : [B, H, W] noisy/coarse mask with values in {0,...,num_classes-1} or 255 for ignore
        t    : [B] or [B,1] diffusion timestep (normalized, e.g. in [0,1])

    Output:
        logits: [B, num_classes, H, W]
    """
    def __init__(
        self,
        in_ch=3,
        num_classes=6,
        time_dim=64,
        base_ch=128,
        ignore_index=255,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        self.time_embed = TimeEmbedding(time_dim=time_dim)

        
        total_in_ch = in_ch + num_classes + num_classes
        # Encoder
        self.d1 = ResidualBlock(total_in_ch, base_ch, time_dim=time_dim)          # H
        self.p1 = nn.MaxPool2d(2)

        self.d2 = ResidualBlock(base_ch, base_ch * 2, time_dim=time_dim)          # H/2
        self.p2 = nn.MaxPool2d(2)

        self.d3 = ResidualBlock(base_ch * 2, base_ch * 4, time_dim=time_dim)      # H/4
        self.attn3 = SelfAttention(base_ch * 4)
        self.p3 = nn.MaxPool2d(2)

        self.d4 = ResidualBlock(base_ch * 4, base_ch * 8, time_dim=time_dim)      # H/8
        self.attn4 = SelfAttention(base_ch * 8)

        # Decoder
        self.u3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.c3 = ResidualBlock(base_ch * 8, base_ch * 4, time_dim=time_dim)

        self.u2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.c2 = ResidualBlock(base_ch * 4, base_ch * 2, time_dim=time_dim)

        self.u1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.c1 = ResidualBlock(base_ch * 2, base_ch, time_dim=time_dim)

        self.outc = nn.Conv2d(base_ch, num_classes, kernel_size=1)
    
    def _mask_to_onehot(self, mask):
        """
        mask: [B,H,W], ignore=255
        return: [B,num_classes,H,W]
        """
        mask = mask.clone().long()
        valid = (mask != self.ignore_index)

        safe_mask = mask.clone()
        safe_mask[~valid] = 0
        safe_mask = safe_mask.clamp(0, self.num_classes - 1)

        onehot = F.one_hot(safe_mask, num_classes=self.num_classes)  # [B,H,W,K]
        onehot = onehot.permute(0, 3, 1, 2).float()                  # [B,K,H,W]
        onehot = onehot * valid.unsqueeze(1).float()

        return onehot



    def forward(self, img, coarse, xt, t):
        if img.dim() != 4:
            raise ValueError(f"img must have shape [B,3,H,W], got {img.shape}")
        if coarse.dim() != 3:
            raise ValueError(f"coarse must have shape [B,H,W], got {coarse.shape}")
        if xt.dim() != 3:
            raise ValueError(f"xt must have shape [B,H,W], got {xt.shape}")

        coarse_oh = self._mask_to_onehot(coarse)   # [B,K,H,W]
        xt_oh = self._mask_to_onehot(xt)           # [B,K,H,W]
        t_emb = self.time_embed(t)

        x = torch.cat([img, coarse_oh, xt_oh], dim=1)



        x1 = self.d1(x, t_emb)
        x2 = self.d2(self.p1(x1), t_emb)
        x3 = self.d3(self.p2(x2), t_emb)
        x3 = self.attn3(x3)
        x4 = self.d4(self.p3(x3), t_emb)
        x4 = self.attn4(x4)

        up3 = self.u3(x4)
        if up3.shape[-2:] != x3.shape[-2:]:
            up3 = F.interpolate(up3, size=x3.shape[-2:], mode="bilinear", align_corners=False)
        up3 = self.c3(torch.cat([up3, x3], dim=1), t_emb)

        up2 = self.u2(up3)
        if up2.shape[-2:] != x2.shape[-2:]:
            up2 = F.interpolate(up2, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        up2 = self.c2(torch.cat([up2, x2], dim=1), t_emb)

        up1 = self.u1(up2)
        if up1.shape[-2:] != x1.shape[-2:]:
            up1 = F.interpolate(up1, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        up1 = self.c1(torch.cat([up1, x1], dim=1), t_emb)

        residual_logits = self.outc(up1)
        logits = self.outc(up1) + xt_oh
        return logits