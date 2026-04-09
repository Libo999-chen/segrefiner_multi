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
    mask: [H, W] with values in {0, ..., K-1, 255}
    """
    x = mask.clone()

    for _ in range(t):
        jump = (torch.rand_like(x.float()) < lam).long()
        valid = (x != 255)
        x[valid] = (x[valid] + jump[valid]) % K

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

    valid = (x0 != 255) & (xt != 255)

    # for invalid pixels, fill with 0 temporarily
    x0v = x0.clone()
    xtv = xt.clone()
    x0v[~valid] = 0
    xtv[~valid] = 0

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

    q_stay[~valid] = 0.0
    q_jump[~valid] = 0.0
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
    xtv = xt.clone()
    valid = (xtv != 255)
    xtv[~valid] = 0

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

    p_stay[~valid] = 0.0
    p_jump[~valid] = 0.0

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
# Evaluation metrics
# ================================================================
def IoU_numpy(pred, gt, num_classes=6, ignore_index=255):
    """
    Compute mean IoU over classes present in gt/pred, ignoring ignore_index.
    pred, gt: [H, W] numpy arrays
    """
    valid = (gt != ignore_index)

    pred = pred[valid]
    gt = gt[valid]

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


def extract_boundary(mask, ignore_index=255):
    """
    Extract semantic boundaries from a label mask.
    A pixel is boundary if any 4-neighbor has a different valid label.
    Returns binary boundary map of shape [H, W], dtype uint8.
    """
    H, W = mask.shape
    boundary = np.zeros((H, W), dtype=np.uint8)

    valid = (mask != ignore_index)

    # up/down differences
    diff_down = (
        valid[:-1, :] & valid[1:, :] &
        (mask[:-1, :] != mask[1:, :])
    )
    boundary[:-1, :] |= diff_down.astype(np.uint8)
    boundary[1:, :]  |= diff_down.astype(np.uint8)

    # left/right differences
    diff_right = (
        valid[:, :-1] & valid[:, 1:] &
        (mask[:, :-1] != mask[:, 1:])
    )
    boundary[:, :-1] |= diff_right.astype(np.uint8)
    boundary[:, 1:]  |= diff_right.astype(np.uint8)

    return boundary


def BFScore(pred, gt, ignore_index=255, tolerance=2):
    """
    Boundary F-score with pixel tolerance.
    pred, gt: [H, W] numpy arrays

    A predicted boundary pixel is correct if it lies within `tolerance`
    pixels of a ground-truth boundary pixel, and vice versa.
    """
    pred_b = extract_boundary(pred, ignore_index=ignore_index)
    gt_b = extract_boundary(gt, ignore_index=ignore_index)

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

    # precision: fraction of predicted boundary matched by GT boundary
    pred_match = (pred_b > 0) & (gt_dil > 0)
    precision = pred_match.sum() / (pred_b.sum() + 1e-8)

    # recall: fraction of GT boundary matched by predicted boundary
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
    split="train"
)
val_dataset = Cityscapes6ClassRefinement(
    root="/home/lc2762/segrefiner_multi/data",
    split="val"
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

model = UNet_Diffusion(num_classes=6).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

K = 6
T = 16
lam = 0.3

# 推荐先用 CE + 小权重 KL，更稳定
alpha_ce = 1.0
alpha_kl = 0.01


# ================================================================
# TRAINING
# ================================================================
for epoch in range(100):
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_kl = 0.0

    for img, coarse, gt in train_loader:
        img = img.to(device)
        coarse = coarse.to(device)
        gt = gt.to(device)

        B = img.size(0)

        # ---- sample timestep ----
        t = torch.randint(1, T + 1, (B,), device=device)

        # ---- forward diffusion from coarse ----
        xt = torch.stack([
            cdd_forward(gt[i], int(t[i]), K, lam)
            for i in range(B)
        ])

        # ---- normalize time ----
        t_norm = t.float() / T

        # ---- model forward ----
        
        logits = model(img, coarse, xt, t_norm)

        # ---- L0: direct x0 reconstruction ----
        ce_loss = F.cross_entropy(logits, gt, ignore_index=255)

        # ---- Lt-1: reverse posterior KL ----
        kl_loss = posterior_kl_loss(logits, gt, xt, t, K=K, lam=lam)

        loss = alpha_ce * ce_loss + alpha_kl * kl_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()
        total_ce += ce_loss.item()
        total_kl += kl_loss.item()

    print(
        f"[Epoch {epoch}] "
        f"loss = {total_loss / len(train_loader):.6f}, "
        f"ce = {total_ce / len(train_loader):.6f}, "
        f"kl = {total_kl / len(train_loader):.6f}"
    )
    # ===== save every 10 epochs =====
    if (epoch + 1) % 50 == 0:
        torch.save(model.state_dict(), f"/home/lc2762/segrefiner_multi/runs/checkpoints/model_epoch_{epoch+1}__akl{alpha_ce}.pth")
        print(f"✅ Saved model at epoch {epoch+1}")


# ================================================================
# EVALUATION
# ================================================================
model.eval()

all_IoU, all_BF = [], []

with torch.no_grad():
    for img, coarse, gt in val_loader:
        img = img.to(device)
        coarse = coarse.to(device)
        gt = gt.to(device)

        B = img.size(0)

        xt = coarse.clone()
        t = torch.ones(B, device=device, dtype=torch.long)
        logits = model(img, coarse, xt, t.float() / T)
        


        pred = torch.argmax(logits, dim=1).cpu().numpy()
        gt_np = gt.cpu().numpy()

        for i in range(B):
            all_IoU.append(IoU_numpy(pred[i], gt_np[i]))
            all_BF.append(BFScore(pred[i], gt_np[i]))

print("\n==== RESULTS ====")
print(f"IoU ↑     {np.mean(all_IoU):.6f}")
print(f"BFScore ↑ {np.mean(all_BF):.6f}")