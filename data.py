import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import cv2


def map_to_6_classes(mask, idx=None):
    new_mask = np.ones_like(mask, dtype=np.uint8) * 5  # other

    # based on Cityscapes trainIds
    mapping = {
        7: 0,   # road
        8: 1,   # sidewalk
        11: 2,  # building
        21: 3,  # vegetation
        26: 4   # car
    }

    for k, v in mapping.items():
        new_mask[mask == k] = v

    if idx == 0:
        print("after mapping:", np.unique(new_mask))

    return new_mask


def generate_coarse_mask(mask, num_classes=6):
    mask = mask.copy()

    h, w = mask.shape
    small = cv2.resize(mask, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    edges = cv2.Canny(mask.astype(np.uint8), 0, 1)
    kernel = np.ones((5, 5), np.uint8)
    band = cv2.dilate(edges, kernel)

    noise = np.random.randint(0, num_classes, size=mask.shape)
    prob = np.random.rand(*mask.shape)

    corrupt_region = (band > 0) & (prob < 0.3)
    mask[corrupt_region] = noise[corrupt_region]

    return mask


class Cityscapes6ClassRefinement(Dataset):
    def __init__(self, root, split="train", size=(256, 256)):
        self.root = root
        self.split = split
        self.size = size

        self.img_dir = os.path.join(root, "leftImg8bit", split)
        self.gt_dir = os.path.join(root, "gtFine", split)

        self.images, self.masks = [], []

        for city in sorted(os.listdir(self.img_dir)):
            img_folder = os.path.join(self.img_dir, city)
            gt_folder = os.path.join(self.gt_dir, city)

            if not os.path.isdir(img_folder) or not os.path.isdir(gt_folder):
                continue

            for f in sorted(os.listdir(img_folder)):
                if f.endswith("_leftImg8bit.png"):
                    img_path = os.path.join(img_folder, f)
                    
                    mask_name = f.replace("_leftImg8bit.png", "_gtFine_labelIds.png")
                    mask_path = os.path.join(gt_folder, mask_name)

                    if os.path.exists(mask_path):
                        self.images.append(img_path)
                        self.masks.append(mask_path)

        self.image_transform = transforms.Compose([
            transforms.Resize(size, interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.mask_resize = transforms.Resize(size, interpolation=Image.NEAREST)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        mask = np.array(Image.open(self.masks[idx]), dtype=np.uint8)
        #print("unique raw mask:", np.unique(mask))

        mask = map_to_6_classes(mask, idx)

        img = self.image_transform(img)
        mask = Image.fromarray(mask)
        mask = self.mask_resize(mask)
        mask = np.array(mask, dtype=np.uint8)

        coarse = generate_coarse_mask(mask)

        mask = torch.from_numpy(mask).long()
        coarse = torch.from_numpy(coarse).long()

        return img, coarse, mask