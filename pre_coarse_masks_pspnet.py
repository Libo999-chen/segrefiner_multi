"""
Generate PSPNet Cityscapes coarse masks. Separate from pre_coarse_masks.py
because there is no first-class Cityscapes-pretrained PSPNet on HuggingFace
— this uses the pretrained checkpoints from the original PSPNet author's repo
hszhao/semseg:

    https://github.com/hszhao/semseg

One-time setup:
    git clone https://github.com/hszhao/semseg.git \\
        /home/lc2762/segrefiner_multi/third_party/semseg

Then download a Cityscapes PSPNet checkpoint from the README's Google Drive
links (under "Performance" -> Cityscapes), e.g.:
    train_epoch_200.pth  for cityscapes_pspnet50  (layers=50)
    train_epoch_200.pth  for cityscapes_pspnet101 (layers=101)

Output: data/coarseMask_pspnet_<backbone>/{train,val}/<city>/<frame>_coarse6.png
in the 6-class scheme used in data.py:
    0 road, 1 sidewalk, 2 building, 3 vegetation, 4 car, 5 other

Re-running is safe: existing outputs are skipped.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm


# Cityscapes 19-trainId -> our 6-class scheme. Unknowns -> "other" (5).
TRAINID_TO_6 = np.full(256, 5, dtype=np.uint8)
TRAINID_TO_6[0] = 0   # road
TRAINID_TO_6[1] = 1   # sidewalk
TRAINID_TO_6[2] = 2   # building
TRAINID_TO_6[8] = 3   # vegetation
TRAINID_TO_6[13] = 4  # car


PREPROCESS = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/home/lc2762/segrefiner_multi/data")
    p.add_argument(
        "--repo-path",
        default="/home/lc2762/segrefiner_multi/third_party/semseg",
        help="local clone of github.com/hszhao/semseg",
    )
    p.add_argument(
        "--ckpt",
        required=True,
        help="path to a cityscapes PSPNet checkpoint (e.g. train_epoch_200.pth).",
    )
    p.add_argument(
        "--layers",
        type=int,
        choices=[50, 101],
        default=50,
        help="ResNet backbone depth used by the checkpoint.",
    )
    p.add_argument(
        "--out-dirname",
        default=None,
        help="auto-derived from --layers if omitted.",
    )
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=[1025, 2049],
        metavar=("H", "W"),
        help="forward-pass resolution. PSPNet (zoom_factor=8) requires "
             "(H-1) %% 8 == 0 and (W-1) %% 8 == 0; native Cityscapes is "
             "1024x2048, so default pads by 1px on each axis.",
    )
    return p.parse_args()


def load_model(repo_path, ckpt, layers, device):
    if not os.path.isdir(repo_path):
        raise FileNotFoundError(
            f"{repo_path} not found. Clone "
            "https://github.com/hszhao/semseg there first."
        )
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(
            f"{ckpt} not found. Download from the hszhao/semseg README "
            "(Cityscapes pretrained models)."
        )

    sys.path.insert(0, repo_path)
    # semseg's PSPNet pulls a torchvision-style ResNet from model/resnet.py;
    # pretrained=False because the checkpoint already includes the backbone.
    from model.pspnet import PSPNet

    model = PSPNet(
        layers=layers,
        classes=19,
        zoom_factor=8,
        pretrained=False,
    )
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = state.get("state_dict", state.get("model_state", state))
    # Strip "module." prefix if present (DataParallel-saved checkpoints).
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    # strict=False: aux head / criterion buffers in the checkpoint are fine to drop.
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model


@torch.no_grad()
def predict_6class(img: Image.Image, model, device, input_size):
    W0, H0 = img.size
    H, W = input_size
    if (H, W) != (H0, W0):
        img = img.resize((W, H), Image.BILINEAR)
    t = PREPROCESS(img).unsqueeze(0).to(device)
    out = model(t)
    # semseg PSPNet returns (main, aux) in training mode, just main in eval.
    logits = out[0] if isinstance(out, (tuple, list)) else out
    if logits.shape[-2:] != (H0, W0):
        logits = F.interpolate(
            logits, size=(H0, W0), mode="bilinear", align_corners=False
        )
    seg = logits.argmax(dim=1)[0].cpu().numpy().astype(np.int64)
    return TRAINID_TO_6[seg]


def collect_jobs(data_root, splits, out_dirname):
    jobs = []
    for split in splits:
        img_root = os.path.join(data_root, "leftImg8bit", split)
        out_root = os.path.join(data_root, out_dirname, split)
        if not os.path.isdir(img_root):
            print(f"[skip] {img_root} not found")
            continue
        for city in sorted(os.listdir(img_root)):
            img_folder = os.path.join(img_root, city)
            if not os.path.isdir(img_folder):
                continue
            out_folder = os.path.join(out_root, city)
            os.makedirs(out_folder, exist_ok=True)
            for f in sorted(os.listdir(img_folder)):
                if not f.endswith("_leftImg8bit.png"):
                    continue
                jobs.append((
                    os.path.join(img_folder, f),
                    os.path.join(out_folder,
                                 f.replace("_leftImg8bit.png", "_coarse6.png")),
                ))
    return jobs


def main():
    args = parse_args()
    out_dirname = args.out_dirname or f"coarseMask_pspnet_{args.layers}"
    H, W = args.input_size
    if (H - 1) % 8 != 0 or (W - 1) % 8 != 0:
        raise ValueError(
            f"--input-size {H} {W} invalid: PSPNet requires (H-1)%8==0 and "
            f"(W-1)%8==0. Try 713 713, 1025 2049, or 713 1281."
        )
    print(f"loading PSPNet (resnet{args.layers}) on {args.device}")
    print(f"  repo: {args.repo_path}")
    print(f"  ckpt: {args.ckpt}")
    print(f"  writing to {os.path.join(args.data_root, out_dirname)}")
    model = load_model(args.repo_path, args.ckpt, args.layers, args.device)

    jobs = collect_jobs(args.data_root, args.splits, out_dirname)
    todo = [(i, o) for i, o in jobs if not os.path.exists(o)]
    print(f"{len(jobs)} total images, {len(todo)} to process")

    for img_path, out_path in tqdm(todo):
        img = Image.open(img_path).convert("RGB")
        seg6 = predict_6class(img, model, args.device, tuple(args.input_size))
        Image.fromarray(seg6.astype(np.uint8)).save(out_path)


if __name__ == "__main__":
    main()
