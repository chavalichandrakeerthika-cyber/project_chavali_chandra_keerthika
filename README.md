# Landslide Detection from Satellite Imagery

**Student:** Keerthika Chavali  
**Roll Number:** 20231073  
**Course:** Image and Video Processing with Deep Learning

---
## Project Overview
This project implements a U-Net Convolutional Neural Network to detect and delineate landslide boundaries from multi-spectral satellite imagery. The model takes 128x128 pixel patches with 14 spectral bands as input and produces binary segmentation masks classifying each pixel as landslide or background.

---
## Setup

### Clone the repository
```bash
git clone https://github.com/chavalichandrakeerthika-cyber/project_chavali_chandra_keerthika.git
cd project_chavali_chandra_keerthika
```

### Install dependencies
```bash
pip install torch h5py numpy matplotlib
```
or
```bash
python -m pip install torch h5py numpy
```

---
## Testing Imports

```bash
python -c "from interface import TheModel, the_trainer, the_predictor, TheDataset, the_dataloader, the_batch_size, total_epochs; print('All imports OK')"
```

---
## Running Inference on Sample Data
Sample data is included in the data/ folder (10 images):

```bash
python -c "
from interface import the_predictor
import os
files = [os.path.join('data', f) for f in os.listdir('data')]
preds = the_predictor(files)
print('Predictions:', len(preds))
print('Mask shape:', preds[0].shape)
"
```
Prediction PNG files will be saved in the project folder.

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

---

