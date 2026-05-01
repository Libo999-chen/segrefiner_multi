"""
Run a Cityscapes-trained semantic segmentation model on every Cityscapes
leftImg8bit image and save a 6-class coarse mask alongside the dataset:

    data/<out-dirname>/{train,val}/<city>/<frame>_coarse6.png

Each saved PNG is a single-channel uint8 map with values in {0..5} using the
"bad_order" 6-class scheme (different class index assignment from data.py):

    0 road, 1 vegetation, 2 sidewalk, 3 car, 4 building, 5 other

Supports two HuggingFace model families that emit the standard 19 Cityscapes
trainIds, dispatched automatically from the checkpoint's model_type:

    facebook/mask2former-*-cityscapes-semantic
    nvidia/segformer-b{0..5}-finetuned-cityscapes-1024-1024

Re-running is safe: existing outputs are skipped.
"""

import argparse
import os
import re

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    Mask2FormerForUniversalSegmentation,
    SegformerForSemanticSegmentation,
)


# Both model families output 19 Cityscapes trainIds.
# Build a 256-entry LUT so unknown ids fall through to "other" (5).
# bad_order mapping: keys below are Cityscapes labelIds, translated to trainIds.
#   labelId 7 (road) -> 0,  21 (vegetation) -> 1,  8 (sidewalk) -> 2,
#   26 (car) -> 3,  11 (building) -> 4,  other -> 5.
TRAINID_TO_6 = np.full(256, 5, dtype=np.uint8)
TRAINID_TO_6[0] = 0   # road        (labelId 7)
TRAINID_TO_6[8] = 1   # vegetation  (labelId 21)
TRAINID_TO_6[1] = 2   # sidewalk    (labelId 8)
TRAINID_TO_6[13] = 3  # car         (labelId 26)
TRAINID_TO_6[2] = 4   # building    (labelId 11)


MODEL_CLASSES = {
    "mask2former": Mask2FormerForUniversalSegmentation,
    "segformer": SegformerForSemanticSegmentation,
}


def default_out_dirname(model_name: str) -> str:
    n = model_name.lower()
    if "mask2former" in n:
        return "coarseMask_m2f_badorder"
    m = re.search(r"segformer-(b\d)", n)
    if m:
        return f"coarseMask_segformer_{m.group(1)}_badorder"
    return "coarseMask_badorder"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/home/lc2762/segrefiner_multi/data")
    p.add_argument(
        "--out-dirname",
        default=None,
        help="written under <data-root>/<out-dirname>/<split>/...; "
             "auto-derived from --model if omitted.",
    )
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument(
        "--model",
        default="facebook/mask2former-swin-large-cityscapes-semantic",
        help="HF checkpoint id. Examples: "
             "facebook/mask2former-swin-large-cityscapes-semantic, "
             "nvidia/segformer-b0-finetuned-cityscapes-1024-1024.",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--fp16", action="store_true",
                   help="Run the model in fp16 on GPU for ~2x speedup.")
    return p.parse_args()


def load_model(name, device, fp16):
    cfg = AutoConfig.from_pretrained(name)
    cls = MODEL_CLASSES.get(cfg.model_type)
    if cls is None:
        raise ValueError(
            f"unsupported model_type='{cfg.model_type}' for {name}; "
            f"add it to MODEL_CLASSES."
        )
    proc = AutoImageProcessor.from_pretrained(name)
    dtype = torch.float16 if (fp16 and device.startswith("cuda")) else torch.float32
    model = cls.from_pretrained(name, torch_dtype=dtype).to(device).eval()
    return proc, model, dtype


@torch.no_grad()
def predict_6class(img: Image.Image, proc, model, device, dtype) -> np.ndarray:
    inputs = proc(images=img, return_tensors="pt")
    inputs = {k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
              for k, v in inputs.items()}
    outputs = model(**inputs)
    seg = proc.post_process_semantic_segmentation(
        outputs, target_sizes=[img.size[::-1]]
    )[0].cpu().numpy().astype(np.int64)
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
    out_dirname = args.out_dirname or default_out_dirname(args.model)
    print(f"loading {args.model} on {args.device} (fp16={args.fp16})")
    print(f"writing to {os.path.join(args.data_root, out_dirname)}")
    proc, model, dtype = load_model(args.model, args.device, args.fp16)

    jobs = collect_jobs(args.data_root, args.splits, out_dirname)
    todo = [(i, o) for i, o in jobs if not os.path.exists(o)]
    print(f"{len(jobs)} total images, {len(todo)} to process")

    for img_path, out_path in tqdm(todo):
        img = Image.open(img_path).convert("RGB")
        seg6 = predict_6class(img, proc, model, args.device, dtype)
        Image.fromarray(seg6.astype(np.uint8)).save(out_path)


if __name__ == "__main__":
    main()
