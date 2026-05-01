"""
Reload a trained checkpoint and re-evaluate on Cityscapes val with
xt = coarse.clone() as the reverse-process initialization.

Also reports a coarse-only baseline (no model) for sanity-check.
"""

import math
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
import cv2

from data import Cityscapes6ClassRefinement
from model import UNet_Diffusion


# ----------------------------------------------------------------------
# Reverse-process pieces (copied from train.py to avoid importing it,
# since train.py runs training at module load time)
# ----------------------------------------------------------------------
def binom_mod_probs(num_steps, K, lam, device):
    probs = torch.zeros(K, device=device, dtype=torch.float32)
    if num_steps == 0:
        probs[0] = 1.0
        return probs
    for j in range(num_steps + 1):
        p = math.comb(num_steps, j) * (lam ** j) * ((1 - lam) ** (num_steps - j))
        probs[j % K] += p
    return probs


def predicted_two_point_posterior_from_x0_logits(logits, xt, t_scalar, K, lam):
    device = logits.device
    probs_tm1 = binom_mod_probs(t_scalar - 1, K, lam, device)

    px0 = F.softmax(logits, dim=0)
    H, W = xt.shape

    p_stay = torch.zeros((H, W), device=device, dtype=torch.float32)
    p_jump = torch.zeros((H, W), device=device, dtype=torch.float32)

    xt_minus_1 = (xt - 1) % K
    for a in range(K):
        q_prev_to_xt = probs_tm1[(xt - a) % K]
        q_prev_to_xt_minus_1 = probs_tm1[(xt_minus_1 - a) % K]
        denom = (1 - lam) * q_prev_to_xt + lam * q_prev_to_xt_minus_1 + 1e-12
        w_stay = ((1 - lam) * q_prev_to_xt) / denom
        w_jump = (lam * q_prev_to_xt_minus_1) / denom
        p_stay += px0[a] * w_stay
        p_jump += px0[a] * w_jump

    s = p_stay + p_jump + 1e-12
    return p_stay / s, p_jump / s


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


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
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
            continue
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

def cdd_forward(mask, t, K=6, lam=0.3):
    """
    mask: [H, W] with values in {0, ..., K-1}
    """
    x = mask.clone()

    for _ in range(t):
        jump = (torch.rand_like(x.float()) < lam).long()
        x = (x + jump) % K

    return x
# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt",
        default="/home/lc2762/segrefiner_multi/runs/checkpoints/dspm_uniform_mask2fwd_best__akl0.001.pth",
    )
    p.add_argument("--data-root", default="/home/lc2762/segrefiner_multi/data")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--T", type=int, default=16)
    p.add_argument("--K", type=int, default=6)
    p.add_argument("--lam", type=float, default=0.3)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    val_dataset = Cityscapes6ClassRefinement(root=args.data_root, split="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    print(f"val samples: {len(val_dataset)}")

    model = UNet_Diffusion(num_classes=args.K).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"loaded checkpoint: {args.ckpt}")

    K = args.K
    coarse_inter,  coarse_union  = np.zeros(K, np.int64), np.zeros(K, np.int64)
    x0_inter,      x0_union      = np.zeros(K, np.int64), np.zeros(K, np.int64)
    refined_inter, refined_union = np.zeros(K, np.int64), np.zeros(K, np.int64)
    coarse_bf, x0_bf, refined_bf = [], [], []
    total_changed_px = 0
    total_px = 0

    with torch.no_grad():
        for img, coarse, gt in val_loader:
            img = img.to(device)
            coarse = coarse.to(device)
            gt = gt.to(device)
            B = img.size(0)

            # ---- coarse-only baseline (no model) ----
            coarse_np = coarse.cpu().numpy()
            gt_np = gt.cpu().numpy()
            for i in range(B):
                iou_update(coarse_inter, coarse_union, coarse_np[i], gt_np[i], num_classes=K)
                coarse_bf.append(BFScore(coarse_np[i], gt_np[i], num_classes=K))

            # ---- (a) direct x_0 prediction via argmax(logits) ----
            # one forward pass at t=T with xt=coarse, skip reverse loop
            t_full = torch.full((B,), args.T, device=device, dtype=torch.long)
            logits_x0 = model(img, coarse, coarse, t_full.float() / args.T)
            x0_pred = logits_x0.argmax(dim=1).cpu().numpy()
            for i in range(B):
                iou_update(x0_inter, x0_union, x0_pred[i], gt_np[i], num_classes=K)
                x0_bf.append(BFScore(x0_pred[i], gt_np[i], num_classes=K))


            #xt = cdd_forward(coarse, t=t_start, K=args.K, lam=args.lam)

            #xt = cdd_forward(coarse, t=args.T//2, K=args.K, lam=args.lam)   # ---- refinement: start f
            xt = torch.randint(0, 6, coarse.shape, device=device)
            #xt = coarse.clone()

            for step in range(args.T, 0, -1):
                t = torch.full((B,), step, device=device, dtype=torch.long)
                logits = model(img, coarse, xt, t.float() / args.T)
                xt = reverse_one_step(logits, xt, step, K=args.K, lam=args.lam)

            # ---- (b) how many pixels did reverse actually flip? ----
            total_changed_px += (xt != coarse).sum().item()
            total_px += xt.numel()

            pred_np = xt.cpu().numpy()
            for i in range(B):
                iou_update(refined_inter, refined_union, pred_np[i], gt_np[i], num_classes=K)
                refined_bf.append(BFScore(pred_np[i], gt_np[i], num_classes=K))

    coarse_iou  = iou_reduce(coarse_inter,  coarse_union)
    x0_iou      = iou_reduce(x0_inter,      x0_union)
    refined_iou = iou_reduce(refined_inter, refined_union)
    coarse_bf_m  = float(np.mean(coarse_bf))
    x0_bf_m      = float(np.mean(x0_bf))
    refined_bf_m = float(np.mean(refined_bf))

    print("\n==== RESULTS (xt = cdd_forward(coarse, t=args.T, K=args.K, lam=args.lam)) ====")
    print(f"{'':12s}  {'IoU':>10s}  {'BFScore':>10s}")
    print(f"{'coarse  ':12s}  {coarse_iou:>10.6f}  {coarse_bf_m:>10.6f}")
    print(f"{'x0_argmax':12s}  {x0_iou:>10.6f}  {x0_bf_m:>10.6f}")
    print(f"{'refined ':12s}  {refined_iou:>10.6f}  {refined_bf_m:>10.6f}")
    print(f"{'delta(ref-coarse)':12s}  {refined_iou-coarse_iou:>+10.6f}  "
          f"{refined_bf_m-coarse_bf_m:>+10.6f}")
    print(f"{'delta(x0-coarse) ':12s}  {x0_iou-coarse_iou:>+10.6f}  "
          f"{x0_bf_m-coarse_bf_m:>+10.6f}")
    print(f"\nrefined changed {100*total_changed_px/total_px:.2f}% of pixels vs coarse")


if __name__ == "__main__":
    main()
