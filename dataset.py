# dataset.py

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import h5py
import numpy as np
import os
from config import batch_size

class LandslideDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.ids = sorted(
            [f.replace("image_", "").replace(".h5", "") for f in os.listdir(img_dir)],
            key=lambda x: int(x)
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        num = self.ids[idx]

        img_path = os.path.join(self.img_dir, f"image_{num}.h5")
        with h5py.File(img_path, 'r') as f:
            img = f['img'][:].astype(np.float32)

        mask_path = os.path.join(self.mask_dir, f"mask_{num}.h5")
        with h5py.File(mask_path, 'r') as f:
            mask = f['mask'][:].astype(np.float32)

        for c in range(img.shape[2]):
            ch = img[:, :, c]
            ch_min, ch_max = ch.min(), ch.max()
            if ch_max - ch_min > 0:
                img[:, :, c] = (ch - ch_min) / (ch_max - ch_min)

        img  = torch.tensor(img).permute(2, 0, 1)
        mask = torch.tensor(mask).unsqueeze(0)
        return img, mask


def get_dataloaders(train_img_dir, train_mask_dir):
    full_dataset = LandslideDataset(train_img_dir, train_mask_dir)
    train_size   = int(0.85 * len(full_dataset))
    val_size     = len(full_dataset) - train_size

    train_dataset, valid_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, valid_loader
