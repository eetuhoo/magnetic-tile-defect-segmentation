import cv2
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


class MetalDefectDataset(Dataset):
    def __init__(self, root_dir=None, parent_dir=None, image_paths=None):
        if image_paths is not None:
            self.images = image_paths
        else:
            self.root_dir = Path(parent_dir, root_dir)
            self.images = sorted(self.root_dir.glob("*.jpg"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = img_path.with_name(img_path.stem + ".png")

        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        #image = image / 255.0

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        #mask = mask / 255.0
        mask = (mask > 0).astype("uint8")

        return image.astype(np.float32), mask.astype(np.float32), str(img_path)
