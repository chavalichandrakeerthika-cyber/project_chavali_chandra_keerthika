# predict.py

import torch
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
from model import UNet
from config import input_channels, output_channels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = UNet(in_channels=input_channels, out_channels=output_channels).to(device)
model.load_state_dict(torch.load(
    os.path.join(BASE_DIR, "checkpoints", "final_weights.pth"),
    map_location=device))
model.eval()


def predict(list_of_img_paths):
    predictions = []

    for img_path in list_of_img_paths:
        with h5py.File(img_path, 'r') as f:
            img = f['img'][:].astype(np.float32)

        for c in range(img.shape[2]):
            ch = img[:, :, c]
            ch_min, ch_max = ch.min(), ch.max()
            if ch_max - ch_min > 0:
                img[:, :, c] = (ch - ch_min) / (ch_max - ch_min)

        img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            output    = model(img_tensor)
            pred_mask = (torch.sigmoid(output) > 0.5).float()
            pred_mask = pred_mask.squeeze().cpu().numpy()

        # Visualize and save
        rgb = img[:, :, :3]
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(rgb)
        axes[0].set_title("Satellite Image")
        axes[0].axis('off')
        axes[1].imshow(pred_mask, cmap='Reds')
        axes[1].set_title(f"Predicted Mask ({pred_mask.mean()*100:.1f}% landslide)")
        axes[1].axis('off')

        filename = os.path.splitext(os.path.basename(img_path))[0]
        plt.suptitle(filename)
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, f"prediction_{filename}.png"), dpi=150)
        plt.close()

        predictions.append(pred_mask)

    print(f"Predictions complete — {len(predictions)} masks generated and saved as PNGs")
    return predictions
