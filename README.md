# Landslide Detection from Satellite Imagery

**Student:** Keerthika Chavali  
**Roll Number:** 20231073  
**Course:** Image and Video Processing with Deep Learning

---
## How to Run

### Install Dependencies
```bash
pip install torch h5py numpy matplotlib
```

### Run Inference
```python
from interface import the_predictor
import os

files = [os.path.join('data', f) for f in os.listdir('data')]
predictions = the_predictor(files)
# Prediction PNG files will be saved in the project folder
```

### Run Training
```python
from interface import TheModel, the_trainer, the_dataloader
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TheModel().to(device)
train_loader, valid_loader = the_dataloader('path/to/TrainData/img', 'path/to/TrainData/mask')
the_trainer(model, train_loader, valid_loader, device)
```

## Project Overview
This project implements a U-Net Convolutional Neural Network to detect and delineate landslide boundaries from multi-spectral satellite imagery. The model takes 128x128 pixel patches with 14 spectral bands as input and produces binary segmentation masks classifying each pixel as landslide or background.

---

## Dataset
- **Name:** Landslide4Sense
- **Source:** Kaggle (tekbahadurkshetri/landslide4sense)
- **Input:** 128x128 multi-spectral image patches (RGB + NIR + topographic bands)
- **Labels:** Binary masks (1 = landslide, 0 = background)
- **Training samples:** 3,799 labeled patches

---

## Model Architecture
- **U-Net** with 4 encoder blocks, bottleneck, and 4 decoder blocks
- Skip connections between encoder and decoder
- Input channels: 14
- Output channels: 1 (binary mask)
- Total parameters: 31,049,857

---

## Results
- **Validation Accuracy:** ~98.7%
- **Best Validation IoU:** 0.5666 (Epoch 19)
- **Loss Function:** BCE + Dice Loss
- **Optimizer:** Adam (lr=1e-4)
- **Epochs:** 20

---

## Project Structure
project_keerthika_chavali/
├── checkpoints/
│   └── final_weights.pth    # trained model weights
├── data/                    # 10 sample .h5 image patches
├── config.py                # hyperparameters and image settings
├── dataset.py               # LandslideDataset class and dataloader
├── model.py                 # UNet and DoubleConv architecture
├── train.py                 # training loop and loss functions
├── predict.py               # inference function
└── interface.py             # standardized imports for grading

---
