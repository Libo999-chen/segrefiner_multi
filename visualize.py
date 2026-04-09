import torch
import numpy as np
import matplotlib.pyplot as plt

from data import Cityscapes6ClassRefinement
from model import UNet_Diffusion

# ================================================================
# Device
# ================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

# ================================================================
# Paths and parameters
# ================================================================
ckpt_path = "/home/lc2762/segrefiner_multi/runs/checkpoints/model_epoch_20__akl1.0.pth"
data_root = "/home/lc2762/segrefiner_multi/data"

K = 6       # number of classes
T = 16      # diffusion steps

# ================================================================
# Cityscapes color palette for the selected 6 classes
# ================================================================
# Class order:
# 0: road, 1: sidewalk, 2: building, 3: vegetation, 4: sky, 5: car
CITYSCAPES_COLORS = np.array([
    [128, 64, 128],   # road
    [244, 35, 232],   # sidewalk
    [70, 70, 70],     # building
    [107, 142, 35],   # vegetation
    [70, 130, 180],   # sky
    [0, 0, 142],      # car
    [0, 0, 0],        # ignore (255)
], dtype=np.uint8)


def decode_segmap(mask, colors=CITYSCAPES_COLORS):
    """
    Convert a label mask to an RGB image using the Cityscapes palette.

    Args:
        mask: [H, W] numpy array with values in {0,...,K-1} and K for ignore.
    Returns:
        color_mask: [H, W, 3] RGB image.
    """
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for label in range(len(colors)):
        color_mask[mask == label] = colors[label]

    return color_mask


# ================================================================
# Load model
# ================================================================
model = UNet_Diffusion(num_classes=K).to(device)

# Use weights_only=True to avoid FutureWarning (if supported by PyTorch version)
try:
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
except TypeError:
    state_dict = torch.load(ckpt_path, map_location=device)

model.load_state_dict(state_dict)
model.eval()

# ================================================================
# Load dataset (validation split)
# ================================================================
dataset = Cityscapes6ClassRefinement(root=data_root, split="val")
print("Dataset size:", len(dataset))
print("First image path:", dataset.images[0])

# Select one sample for visualization
img, coarse, gt = dataset[0]

img = img.unsqueeze(0).to(device)        # [1, 3, H, W]
coarse = coarse.unsqueeze(0).to(device)  # [1, H, W]
gt = gt.unsqueeze(0).to(device)          # [1, H, W]

# Diffusion timestep (consistent with training)
t = torch.ones(1, device=device)
t_norm = t / T
xt = coarse.clone()

# ================================================================
# Inference
# ================================================================
with torch.no_grad():
    logits = model(img, coarse, xt, t_norm)
    pred = torch.argmax(logits, dim=1)

# ================================================================
# Convert tensors to numpy
# ================================================================
# De-normalize image
img_np = img[0].cpu().permute(1, 2, 0).numpy()
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
img_np = img_np * std + mean
img_np = np.clip(img_np, 0, 1)

coarse_np = coarse[0].cpu().numpy()
pred_np = pred[0].cpu().numpy()
gt_np = gt[0].cpu().numpy()

# Map ignore label (255) to color index K
coarse_show = coarse_np.copy()
gt_show = gt_np.copy()
coarse_show[coarse_show == 255] = K
gt_show[gt_show == 255] = K

# Decode segmentation masks to RGB
coarse_rgb = decode_segmap(coarse_show)
pred_rgb = decode_segmap(pred_np)
gt_rgb = decode_segmap(gt_show)

# ================================================================
# Visualization
# ================================================================
plt.figure(figsize=(14, 4))

plt.subplot(1, 4, 1)
plt.imshow(img_np)
plt.title("Image")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(coarse_rgb)
plt.title("Coarse Mask")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(pred_rgb)
plt.title("Prediction")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(gt_rgb)
plt.title("Ground Truth")
plt.axis("off")

plt.tight_layout()
plt.savefig("vis_result.png", dpi=300)
plt.show()

print("Visualization saved to vis_result.png")