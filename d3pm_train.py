import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import cv2

from data import Cityscapes6ClassRefinement
from model import UNet_Diffusion
import os
os.makedirs("/home/lc2762/segrefiner_multi/runs/checkpoints", exist_ok=True)
# ================================================================
# Uniform diffusion forward  q(x_t | x_0) = ᾱ_t·δ(x_t,x_0) + (1-ᾱ_t)/K
# ᾱ_t = (1-beta)^t  (product over t independent identical steps)
# ================================================================
def uniform_forward(mask, t, K=6, beta=0.3):
    """
    mask: [H, W] with values in {0,...,K-1}
    Samples x_t from the closed-form marginal q(x_t | x_0).
    """
    alpha_bar_t = (1.0 - beta) ** t
    keep = torch.rand_like(mask.float()) < alpha_bar_t
    noise = torch.randint(0, K, mask.shape, device=mask.device)
    return torch.where(keep, mask, noise)


# ================================================================
# True reverse posterior  q(x_{t-1} | x_t, x_0)  — full K-class distribution
#
# unnorm[j] = q(x_t | x_{t-1}=j) · q_{t-1}(j | x_0)
#           = (alpha·δ(j,x_t) + beta/K) · (ᾱ_{t-1}·δ(j,x_0) + (1-ᾱ_{t-1})/K)
# ================================================================
def true_full_posterior(x0, xt, t_scalar, K, beta):
    """
    x0, xt: [H, W], values in {0,...,K-1}
    Returns q_prev: [K, H, W] — normalised posterior q(x_{t-1} | x_t, x_0).
    """
    alpha = 1.0 - beta
    alpha_bar_tm1 = alpha ** (t_scalar - 1)

    xt_oh = F.one_hot(xt, K).permute(2, 0, 1).float()   # [K,H,W]
    x0_oh = F.one_hot(x0, K).permute(2, 0, 1).float()   # [K,H,W]

    q_prev = (alpha * xt_oh + beta / K) * (alpha_bar_tm1 * x0_oh + (1.0 - alpha_bar_tm1) / K)
    q_prev = q_prev / (q_prev.sum(dim=0, keepdim=True) + 1e-12)
    return q_prev


# ================================================================
# Predicted reverse posterior  p_theta(x_{t-1} | x_t)
#
# p(x_{t-1}=j | x_t) = Σ_c p_theta(x_0=c) · q(x_{t-1}=j | x_t, x_0=c)
#
# Closed form (derived by substituting the 4-term expansion of q and
# grouping by w_c = p_theta(x_0=c) / q_t(x_t | x_0=c)):
#
#   p_prev[j] = w_j·(α·ᾱ_{t-1}·δ(j,x_t) + β·ᾱ_{t-1}/K)
#             + w_sum·(α·(1-ᾱ_{t-1})/K·δ(j,x_t) + β·(1-ᾱ_{t-1})/K²)
# ================================================================
def predicted_full_posterior_from_x0_logits(logits, xt, t_scalar, K, beta):
    """
    logits: [K, H, W] = model output logits for x_0
    xt    : [H, W]
    Returns p_prev: [K, H, W] — normalised p(x_{t-1} | x_t).
    """
    alpha = 1.0 - beta
    alpha_bar_t  = alpha ** t_scalar
    alpha_bar_tm1 = alpha ** (t_scalar - 1)

    px0   = F.softmax(logits, dim=0)                            # [K, H, W]
    xt_oh = F.one_hot(xt, K).permute(2, 0, 1).float()          # [K, H, W]

    # Z[c] = q_t(x_t | x_0=c)  (marginal used for importance weight)
    Z    = alpha_bar_t * xt_oh + (1.0 - alpha_bar_t) / K       # [K, H, W]
    w    = px0 / (Z + 1e-12)                                    # [K, H, W]
    w_sum = w.sum(dim=0, keepdim=True)                          # [1, H, W]

    p_prev = (w    * (alpha * alpha_bar_tm1 * xt_oh + beta * alpha_bar_tm1 / K)
            + w_sum * (alpha * (1.0 - alpha_bar_tm1) / K * xt_oh
                       + beta * (1.0 - alpha_bar_tm1) / K ** 2))
    p_prev = p_prev / (p_prev.sum(dim=0, keepdim=True) + 1e-12)
    return p_prev


# ================================================================
# KL loss  KL( q(x_{t-1}|x_t,x_0) || p_theta(x_{t-1}|x_t) )
# ================================================================
def posterior_kl_loss(logits, x0, xt, t, K=6, beta=0.3):
    """
    logits: [B, K, H, W]
    x0    : [B, H, W]
    xt    : [B, H, W]
    t     : [B]
    """
    B = logits.size(0)
    total_kl = 0.0
    n_pixels  = 0

    for i in range(B):
        ti = int(t[i].item())
        q = true_full_posterior(x0[i], xt[i], ti, K, beta)                        # [K,H,W]
        p = predicted_full_posterior_from_x0_logits(logits[i], xt[i], ti, K, beta)# [K,H,W]

        # KL(q || p) summed over classes, then summed over pixels
        kl_map = (q * (torch.log(q + 1e-12) - torch.log(p + 1e-12))).sum(dim=0)  # [H,W]
        total_kl += kl_map.sum()
        n_pixels  += kl_map.numel()

    return total_kl / (n_pixels + 1e-12)

# ================================================================
# Boundary-aware texture loss
#   L_tex = alpha * || S(p_theta) - S(x_0) ||_1
#   S(u)  = sqrt( (K_x * f_m(u))^2 + (K_y * f_m(u))^2 + eps )
#   f_m   : 3x3 mean filter,  K_x/K_y: Sobel kernels
# Multi-class extension: applied per-class on softmax probs vs one-hot GT.
# ================================================================
_SOBEL_X = torch.tensor([[-1., 0., 1.],
                         [-2., 0., 2.],
                         [-1., 0., 1.]])
_SOBEL_Y = torch.tensor([[-1., -2., -1.],
                         [ 0.,  0.,  0.],
                         [ 1.,  2.,  1.]])


def boundary_texture_loss(logits, gt, K=6, eps=1e-6):
    """
    logits: [B, K, H, W]
    gt    : [B, H, W] long in {0,...,K-1}
    """
    device = logits.device
    C = logits.size(1)

    probs = F.softmax(logits, dim=1)  # [B, K, H, W]
    gt_oh = F.one_hot(gt.clamp(0, K - 1), num_classes=K).permute(0, 3, 1, 2).float()

    mean_k = torch.ones(C, 1, 3, 3, device=device) / 9.0
    sobel_x = _SOBEL_X.to(device).view(1, 1, 3, 3).expand(C, 1, 3, 3).contiguous()
    sobel_y = _SOBEL_Y.to(device).view(1, 1, 3, 3).expand(C, 1, 3, 3).contiguous()

    p_m = F.conv2d(probs, mean_k, padding=1, groups=C)
    g_m = F.conv2d(gt_oh, mean_k, padding=1, groups=C)

    p_S = torch.sqrt(F.conv2d(p_m, sobel_x, padding=1, groups=C) ** 2 +
                     F.conv2d(p_m, sobel_y, padding=1, groups=C) ** 2 + eps)
    g_S = torch.sqrt(F.conv2d(g_m, sobel_x, padding=1, groups=C) ** 2 +
                     F.conv2d(g_m, sobel_y, padding=1, groups=C) ** 2 + eps)

    return F.l1_loss(p_S, g_S)


# ================================================================
# Evaluation metrics
# ================================================================
def IoU_numpy(pred, gt, num_classes=6):
    """
    Compute mean IoU over classes present in gt/pred, ignoring ignore_index.
    pred, gt: [H, W] numpy arrays
    """


    ious = []
    for c in range(num_classes):
        pred_c = (pred == c)
        gt_c = (gt == c)

        union = np.logical_or(pred_c, gt_c).sum()
        if union == 0:
            continue  # skip absent classes

        inter = np.logical_and(pred_c, gt_c).sum()
        ious.append(inter / (union + 1e-8))

    if len(ious) == 0:
        return 0.0
    return float(np.mean(ious))


def extract_boundary(mask):
    H, W = mask.shape
    boundary = np.zeros((H, W), dtype=np.uint8)

    valid = np.ones_like(mask, dtype=bool)

    diff_down = (
        valid[:-1, :] & valid[1:, :] &
        (mask[:-1, :] != mask[1:, :])
    )
    boundary[:-1, :] |= diff_down.astype(np.uint8)
    boundary[1:, :]  |= diff_down.astype(np.uint8)

    diff_right = (
        valid[:, :-1] & valid[:, 1:] &
        (mask[:, :-1] != mask[:, 1:])
    )
    boundary[:, :-1] |= diff_right.astype(np.uint8)
    boundary[:, 1:]  |= diff_right.astype(np.uint8)

    return boundary


def BFScore(pred, gt, tolerance=2):
    pred_b = extract_boundary(pred)
    gt_b = extract_boundary(gt)

    pred_b = pred_b.astype(np.uint8)
    gt_b = gt_b.astype(np.uint8)

    if pred_b.sum() == 0 and gt_b.sum() == 0:
        return 1.0
    if pred_b.sum() == 0 or gt_b.sum() == 0:
        return 0.0

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * tolerance + 1, 2 * tolerance + 1)
    )

    gt_dil = cv2.dilate(gt_b, kernel)
    pred_dil = cv2.dilate(pred_b, kernel)

    pred_match = (pred_b > 0) & (gt_dil > 0)
    precision = pred_match.sum() / (pred_b.sum() + 1e-8)

    gt_match = (gt_b > 0) & (pred_dil > 0)
    recall = gt_match.sum() / (gt_b.sum() + 1e-8)

    if precision + recall == 0:
        return 0.0

    return float(2 * precision * recall / (precision + recall + 1e-8))


# ================================================================
# Setup
# ================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataset = Cityscapes6ClassRefinement(
    root="/home/lc2762/segrefiner_multi/data",
    split="train",
    coarse_dirname="coarseMask_m2f",
)
val_dataset = Cityscapes6ClassRefinement(
    root="/home/lc2762/segrefiner_multi/data",
    split="val",
    coarse_dirname="coarseMask_m2f",
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

model = UNet_Diffusion(num_classes=6).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

# ---- resume from epoch-30 checkpoint ----
#ckpt_path = "/home/lc2762/segrefiner_multi/runs/checkpoints/dlv3p_mixedxt_model_epoch_30__akl0.001.pth"
#model.load_state_dict(torch.load(ckpt_path, map_location=device))
# start_epoch = 30
# end_epoch = 120
# print(f"✅ Resumed from {ckpt_path}, training {start_epoch} → {end_epoch}")
start_epoch = 0
end_epoch = 120               # 上限，实际由早停决定
print(f"✅ Training from scratch, {start_epoch} → {end_epoch} (with early stopping)")

K = 6
T = 16
beta = 0.3

# 推荐先用 CE + 小权重 KL，更稳定
alpha_ce = 1.0
alpha_kl = 0.001
alpha_tex = 5

# ---- early stopping ----
patience = 15                  # 连续多少个 epoch 没有相对改进就停
min_rel_delta = 1e-3           # 相对下降阈值 (loss_new < best * (1 - min_rel_delta) 才算改进)
best_loss = float("inf")
epochs_since_improve = 0
best_ckpt_path = f"/home/lc2762/segrefiner_multi/runs/checkpoints/dspm_uniform_mask2fwd_best__akl{alpha_kl}.pth"


# ================================================================
# TRAINING
# ================================================================
for epoch in range(start_epoch, end_epoch):
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_kl = 0.0
    total_tex = 0.0

    for img, coarse, gt in train_loader:
        img = img.to(device)
        coarse = coarse.to(device)
        gt = gt.to(device)

        B = img.size(0)

        # ---- sample timestep ----
        t = torch.randint(1, T + 1, (B,), device=device)

        # 50% of samples: xt = coarse with t=T, so the model sees the
        # real test-time input distribution during training.
        use_coarse = torch.rand(B, device=device) < 0.5
        t = torch.where(use_coarse, torch.full_like(t, T), t)

        xt = torch.stack([
            coarse[i].clone() if bool(use_coarse[i]) else uniform_forward(gt[i], int(t[i]), K, beta)
            for i in range(B)
        ])

        # ---- normalize time ----
        t_norm = t.float() / T

        # ---- model forward ----

        logits = model(img, coarse, xt, t_norm)

        # ---- L0: direct x0 reconstruction ----

        ce_per_pixel = F.cross_entropy(logits, gt, reduction='none')
        weight = torch.where(coarse != gt, 4.0, 1.0)
        ce_loss = (ce_per_pixel * weight).mean()

        # ---- Lt-1: reverse posterior KL ----
        kl_loss = posterior_kl_loss(logits, gt, xt, t, K=K, beta=beta)

        # ---- boundary-aware texture loss ----
        tex_loss = boundary_texture_loss(logits, gt, K=K)

        loss = alpha_ce * ce_loss + alpha_kl * kl_loss + alpha_tex * tex_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()
        total_ce += (alpha_ce * ce_loss).item()
        total_kl += (alpha_kl * kl_loss).item()
        total_tex += (alpha_tex * tex_loss).item()

    avg_loss = total_loss / len(train_loader)
    print(
        f"[Epoch {epoch}] "
        f"loss = {avg_loss:.6f}, "
        f"ce = {total_ce / len(train_loader):.6f}, "
        f"kl = {total_kl / len(train_loader):.6f}, "
        f"tex = {total_tex / len(train_loader):.6f}"
    )

    # ---- early stopping & best checkpoint ----
    if avg_loss < best_loss * (1 - min_rel_delta):
        best_loss = avg_loss
        epochs_since_improve = 0
        torch.save(model.state_dict(), best_ckpt_path)
        print(f"  ↳ new best loss {best_loss:.6f}, saved → {best_ckpt_path}")
    else:
        epochs_since_improve += 1
        print(f"  ↳ no improvement ({epochs_since_improve}/{patience}), best = {best_loss:.6f}")

    # ---- periodic checkpoint every 50 epochs ----
    if (epoch + 1) % 50 == 0:
        periodic_ckpt_path = f"/home/lc2762/segrefiner_multi/runs/checkpoints/dspm_uniform_mask2fwd_epoch{epoch + 1}__akl{alpha_kl}.pth"
        torch.save(model.state_dict(), periodic_ckpt_path)
        print(f"  ↳ periodic checkpoint saved → {periodic_ckpt_path}")

    if epochs_since_improve >= patience:
        print(f"⏹  Early stop at epoch {epoch} (no improvement for {patience} epochs). Best loss = {best_loss:.6f}")
        break


# ================================================================
# EVALUATION
# ================================================================

def reverse_one_step(logits, xt, t_scalar, K=6, beta=0.3):
    B, _, H, W = logits.shape
    x_prev = xt.clone()

    for i in range(B):
        p_prev = predicted_full_posterior_from_x0_logits(logits[i], xt[i], t_scalar, K, beta)
        x_prev[i] = p_prev.argmax(dim=0)

    return x_prev

# load best-loss checkpoint for evaluation
if os.path.exists(best_ckpt_path):
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    print(f"✅ Loaded best checkpoint for eval: {best_ckpt_path} (loss={best_loss:.6f})")

model.eval()
all_IoU, all_BF = [], []

with torch.no_grad():
    for img, coarse, gt in val_loader:
        img = img.to(device)
        coarse = coarse.to(device)
        gt = gt.to(device)

        B = img.size(0)

        # x_T sampled from uniform distribution
        xt = torch.randint(0, K, coarse.shape, device=device)


        for step in range(T, 0, -1):
            t = torch.full((B,), step, device=device, dtype=torch.long)
            logits = model(img, coarse, xt, t.float() / T)
            xt = reverse_one_step(logits, xt, step, K=K, beta=beta)

        pred = xt.cpu().numpy()
        gt_np = gt.cpu().numpy()

        for i in range(B):
            all_IoU.append(IoU_numpy(pred[i], gt_np[i]))
            all_BF.append(BFScore(pred[i], gt_np[i]))

print("\n==== RESULTS ====")
print(f"IoU ↑     {np.mean(all_IoU):.6f}")
print(f"BFScore ↑ {np.mean(all_BF):.6f}")