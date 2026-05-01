import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


def map_to_6_classes(mask, idx=None):
    new_mask = np.ones_like(mask, dtype=np.uint8) * 5  # other

    # based on Cityscapes labelIds (from gtFine_labelIds.png)
    mapping = {
        7: 0,    # road
        8: 1,    # sidewalk
        11: 2,   # building
        21: 3,   # vegetation
        26: 4,   # car
        -1: 5,   # other
    }
    # bad order
    # mapping = {
    #     7: 0,    # road
    #     21: 1,   # vegetation
    #     8: 2,    # sidewalk
    #     26: 3,   # car
    #     11: 4,   # building
    #     -1: 5,   # other
    # }

    for k, v in mapping.items():
        new_mask[mask == k] = v

    if idx == 0:
        print("after mapping:", np.unique(new_mask))

    return new_mask


class Cityscapes6ClassRefinement(Dataset):
    def __init__(self, root, split="train", size=(256, 256),
                 coarse_dirname="coarseMask_m2f"):
        self.root = root
        self.split = split
        self.size = size

        self.img_dir = os.path.join(root, "leftImg8bit", split)
        self.gt_dir = os.path.join(root, "gtFine", split)
        self.coarse_dir = os.path.join(root, coarse_dirname, split)

        self.images, self.masks, self.coarses = [], [], []

        for city in sorted(os.listdir(self.img_dir)):
            img_folder = os.path.join(self.img_dir, city)
            gt_folder = os.path.join(self.gt_dir, city)
            coarse_folder = os.path.join(self.coarse_dir, city)

            if not os.path.isdir(img_folder) or not os.path.isdir(gt_folder):
                continue

            for f in sorted(os.listdir(img_folder)):
                if f.endswith("_leftImg8bit.png"):
                    img_path = os.path.join(img_folder, f)
                    mask_path = os.path.join(
                        gt_folder,
                        f.replace("_leftImg8bit.png", "_gtFine_labelIds.png"),
                    )
                    coarse_path = os.path.join(
                        coarse_folder,
                        f.replace("_leftImg8bit.png", "_coarse6.png"),
                    )

                    if os.path.exists(mask_path) and os.path.exists(coarse_path):
                        self.images.append(img_path)
                        self.masks.append(mask_path)
                        self.coarses.append(coarse_path)

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
        coarse = np.array(Image.open(self.coarses[idx]), dtype=np.uint8)

        mask = map_to_6_classes(mask, idx)

        img = self.image_transform(img)

        mask = np.array(self.mask_resize(Image.fromarray(mask)), dtype=np.uint8)
        coarse = np.array(self.mask_resize(Image.fromarray(coarse)), dtype=np.uint8)

        mask = torch.from_numpy(mask).long()
        coarse = torch.from_numpy(coarse).long()

        return img, coarse, mask