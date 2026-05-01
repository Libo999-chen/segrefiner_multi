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
# CDD forward (supports ignore=255)
# ================================================================
def cdd_forward(mask, t, K=6, lam=0.3):
    """
    mask: [H, W] with values in {0, ..., K-1}
    """
    x = mask.clone()

    for _ in range(t):
        jump = (torch.rand_like(x.float()) < lam).long()
        x = (x + jump) % K

    return x


# ================================================================
# Binomial modulo-K transition probabilities
# q_t(x_t | x_0): offset distribution after t cyclic steps
# ================================================================
def binom_mod_probs(num_steps, K, lam, device):
    """
    Return vector probs[r] = P( offset = r mod K ) after num_steps,
    where each step: stay with prob 1-lam, +1 mod K with prob lam.
    """
    probs = torch.zeros(K, device=device, dtype=torch.float32)

    if num_steps == 0:
        probs[0] = 1.0
        return probs

    for j in range(num_steps + 1):
        p = math.comb(num_steps, j) * (lam ** j) * ((1 - lam) ** (num_steps - j))
        probs[j % K] += p

    return probs


# ================================================================
# True posterior q(x_{t-1} | x_t, x_0) for cyclic jump process
# Since one step backward can only be:
#   - stay at x_t
#   - come from x_t - 1
# We return the 2-point posterior:
#   q_stay  = q(x_{t-1}=x_t   | x_t, x_0)
#   q_jump  = q(x_{t-1}=x_t-1 | x_t, x_0)
# ================================================================
def true_two_point_posterior(x0, xt, t_scalar, K, lam):
    """
    x0, xt: [H, W], values in {0,...,K-1,255}
    t_scalar: int in {1,...,T}
    returns:
        q_stay, q_jump, valid_mask  each [H, W]
    """
    device = x0.device
    probs_tm1 = binom_mod_probs(t_scalar - 1, K, lam, device)  # q_{t-1}

    valid = torch.ones_like(x0, dtype=torch.bool)

    x0v = x0
    xtv = xt

    # q_{t-1}(x_t | x_0)
    offset_stay = (xtv - x0v) % K
    q_prev_to_xt = probs_tm1[offset_stay]

    # q_{t-1}(x_t - 1 | x_0)
    xt_minus_1 = (xtv - 1) % K
    offset_jump = (xt_minus_1 - x0v) % K
    q_prev_to_xt_minus_1 = probs_tm1[offset_jump]

    denom = (1 - lam) * q_prev_to_xt + lam * q_prev_to_xt_minus_1
    denom = denom + 1e-12

    q_stay = ((1 - lam) * q_prev_to_xt) / denom
    q_jump = (lam * q_prev_to_xt_minus_1) / denom


    return q_stay, q_jump, valid


# ================================================================
# Predicted reverse posterior induced by p_theta(x0 | xt, img)
# We keep model unchanged: model outputs logits for x0.
# From p_theta(x0 | xt), induce:
#   p_theta(x_{t-1} | x_t)
# again as a 2-point distribution on {x_t, x_t-1}
# ================================================================
def predicted_two_point_posterior_from_x0_logits(logits, xt, t_scalar, K, lam):
    """
    logits: [K, H, W] = model output logits for x0
    xt    : [H, W]
    returns:
        p_stay, p_jump each [H, W]
    """
    device = logits.device
    probs_tm1 = binom_mod_probs(t_scalar - 1, K, lam, device)  # q_{t-1}

    px0 = F.softmax(logits, dim=0)  # [K, H, W]

    H, W = xt.shape
    xtv = xt

    p_stay = torch.zeros((H, W), device=device, dtype=torch.float32)
    p_jump = torch.zeros((H, W), device=device, dtype=torch.float32)

    xt_minus_1 = (xtv - 1) % K

    for a in range(K):
        # q_{t-1}(x_t | x0=a)
        q_prev_to_xt = probs_tm1[(xtv - a) % K]

        # q_{t-1}(x_t - 1 | x0=a)
        q_prev_to_xt_minus_1 = probs_tm1[(xt_minus_1 - a) % K]

        denom = (1 - lam) * q_prev_to_xt + lam * q_prev_to_xt_minus_1
        denom = denom + 1e-12

        w_stay = ((1 - lam) * q_prev_to_xt) / denom
        w_jump = (lam * q_prev_to_xt_minus_1) / denom

        p_stay += px0[a] * w_stay
        p_jump += px0[a] * w_jump


    # numerical cleanup
    s = p_stay + p_jump + 1e-12
    p_stay = p_stay / s
    p_jump = p_jump / s
    return p_stay, p_jump


# ================================================================
# KL loss between true reverse posterior and predicted reverse posterior
# ================================================================
def posterior_kl_loss(logits, x0, xt, t, K=6, lam=0.3):
    """
    logits: [B, K, H, W]
    x0    : [B, H, W]
    xt    : [B, H, W]
    t     : [B]
    """
    B = logits.size(0)
    total_kl = 0.0
    total_valid = 0.0

    for i in range(B):
        ti = int(t[i].item())

        q_stay, q_jump, valid = true_two_point_posterior(x0[i], xt[i], ti, K, lam)
        p_stay, p_jump = predicted_two_point_posterior_from_x0_logits(logits[i], xt[i], ti, K, lam)

        # KL(q || p) = q1 log(q1/p1) + q2 log(q2/p2)
        kl_map = q_stay * (torch.log(q_stay + 1e-12) - torch.log(p_stay + 1e-12)) + \
                 q_jump * (torch.log(q_jump + 1e-12) - torch.log(p_jump + 1e-12))

        total_kl += kl_map[valid].sum()
        total_valid += valid.sum().item()

    return total_kl / (total_valid + 1e-12)

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
def iou_update(inter, union, pred, gt, num_classes=6):
    """Accumulate per-class intersection/union counts for one image."""
    for c in range(num_classes):
        pc = (pred == c)
        gc = (gt == c)
        inter[c] += int(np.logical_and(pc, gc).sum())
        union[c] += int(np.logical_or(pc, gc).sum())


def iou_reduce(inter, union):
    """Mean IoU over classes with union > 0 (dataset-aggregated)."""
    ious = [inter[c] / union[c] for c in range(len(inter)) if union[c] > 0]
    return float(np.mean(ious)) if ious else 0.0


def _class_boundary(mask, c):
    """1-pixel-thick inner boundary of class c: pixels of class c adjacent to non-c."""
    cls = (mask == c).astype(np.uint8)
    if cls.sum() == 0:
        return cls
    not_cls = (1 - cls).astype(np.uint8)
    not_cls_dil = cv2.dilate(not_cls, np.ones((3, 3), np.uint8))
    return (cls & not_cls_dil).astype(np.uint8)


def BFScore(pred, gt, num_classes=6, tolerance=2):
    """Per-class F1 between class-c contours in pred and gt, averaged over present classes."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * tolerance + 1, 2 * tolerance + 1)
    )

    f1s = []
    for c in range(num_classes):
        pb = _class_boundary(pred, c)
        gb = _class_boundary(gt, c)

        ps, gs = pb.sum(), gb.sum()
        if ps == 0 and gs == 0:
            continue  # class absent in both — skip
        if ps == 0 or gs == 0:
            f1s.append(0.0)
            continue

        gb_dil = cv2.dilate(gb, kernel)
        pb_dil = cv2.dilate(pb, kernel)

        precision = ((pb > 0) & (gb_dil > 0)).sum() / (ps + 1e-8)
        recall    = ((gb > 0) & (pb_dil > 0)).sum() / (gs + 1e-8)

        if precision + recall == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * precision * recall / (precision + recall + 1e-8))

    return float(np.mean(f1s)) if f1s else 0.0


# ================================================================
# Setup
# ================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

coarse_dirname = "coarseMask_m2f_badorder"
coarse_tag = coarse_dirname.removeprefix("coarseMask_")

train_dataset = Cityscapes6ClassRefinement(
    root="/home/lc2762/segrefiner_multi/data",
    split="train",
    coarse_dirname=coarse_dirname,
)
val_dataset = Cityscapes6ClassRefinement(
    root="/home/lc2762/segrefiner_multi/data",
    split="val",
    coarse_dirname=coarse_dirname,
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
end_epoch = 120           # 上限，实际由早停决定
print(f"✅ Training from scratch, {start_epoch} → {end_epoch} (with early stopping)")

K = 6
T = 16
lam = 0.3

# 推荐先用 CE + 小权重 KL，更稳定
alpha_ce = 1.0
alpha_kl = 0.001
alpha_tex = 5

# ---- early stopping ----
patience = 15                  # 连续多少个 epoch 没有相对改进就停
min_rel_delta = 1e-3           # 相对下降阈值 (loss_new < best * (1 - min_rel_delta) 才算改进)
best_loss = float("inf")
epochs_since_improve = 0
# best_ckpt_path = f"/home/lc2762/segrefiner_multi/runs/checkpoints/{coarse_tag}_mixedxt_best__akl{alpha_kl}.pth"
best_ckpt_path = f"/home/lc2762/segrefiner_multi/runs/checkpoints/{coarse_tag}_mixedxt_best__akl{alpha_kl}.pth"
save_every = 50


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

        # ---- forward diffusion from gt (x0) ----
        # xt = torch.stack([
        #     cdd_forward(gt[i], int(t[i]), K, lam)
        #     for i in range(B)
        # ])

        # 50% of samples: xt = coarse with t=T, so the model sees the
        # real test-time input distribution during training.
        use_coarse = torch.rand(B, device=device) < 0.5
        t = torch.where(use_coarse, torch.full_like(t, T), t)

        xt = torch.stack([
            coarse[i].clone() if bool(use_coarse[i]) else cdd_forward(gt[i], int(t[i]), K, lam)
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
        kl_loss = posterior_kl_loss(logits, gt, xt, t, K=K, lam=lam)

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

    # ---- periodic checkpoint every `save_every` epochs ----
    if (epoch + 1) % save_every == 0:
        # periodic_path = (
        #     f"/home/lc2762/segrefiner_multi/runs/checkpoints/"
        #     f"{coarse_tag}_mixedxt_epoch_{epoch + 1}__akl{alpha_kl}.pth"
        # )
        periodic_path = (
            f"/home/lc2762/segrefiner_multi/runs/checkpoints/"
            f"{coarse_tag}_mixedxt_epoch_{epoch + 1}__akl{alpha_kl}.pth"
        )
        torch.save(model.state_dict(), periodic_path)
        print(f"  ↳ periodic checkpoint saved → {periodic_path}")

    if epochs_since_improve >= patience:
        print(f"⏹  Early stop at epoch {epoch} (no improvement for {patience} epochs). Best loss = {best_loss:.6f}")
        break


# ================================================================
# EVALUATION
# ================================================================

def reverse_one_step(logits, xt, t_scalar, K=6, lam=0.3):
    B, _, H, W = logits.shape
    x_prev = xt.clone()

    for i in range(B):
        p_stay, p_jump = predicted_two_point_posterior_from_x0_logits(
            logits[i], xt[i], t_scalar, K, lam
        )
        jump = (p_jump > p_stay) 
        x_prev[i][jump] = (xt[i][jump] - 1) % K

    return x_prev

# load best-loss checkpoint for evaluation
if os.path.exists(best_ckpt_path):
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    print(f"✅ Loaded best checkpoint for eval: {best_ckpt_path} (loss={best_loss:.6f})")

model.eval()
inter = np.zeros(K, dtype=np.int64)
union = np.zeros(K, dtype=np.int64)
all_BF = []

with torch.no_grad():
    for img, coarse, gt in val_loader:
        img = img.to(device)
        coarse = coarse.to(device)
        gt = gt.to(device)

        B = img.size(0)

        # refinement: start from coarse
        #xt = coarse.clone()
        xt = torch.randint(0, K, coarse.shape, device=device)  # x_T 从均匀分布采样


        for step in range(T, 0, -1):
            t = torch.full((B,), step, device=device, dtype=torch.long)
            logits = model(img, coarse, xt, t.float() / T)
            xt = reverse_one_step(logits, xt, step, K=K, lam=lam)

        pred = xt.cpu().numpy()
        gt_np = gt.cpu().numpy()

        for i in range(B):
            iou_update(inter, union, pred[i], gt_np[i], num_classes=K)
            all_BF.append(BFScore(pred[i], gt_np[i], num_classes=K))

print("\n==== RESULTS ====")
print(f"IoU ↑     {iou_reduce(inter, union):.6f}")
print(f"BFScore ↑ {np.mean(all_BF):.6f}")