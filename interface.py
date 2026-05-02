# interface.py

# replace MyCustomModel with the name of your model
from model import UNet as TheModel

# the function inside train.py that runs the training loop
from train import train_model as the_trainer

# the function inside predict.py that generates inference
from predict import predict as the_predictor

# your custom Dataset class
from dataset import LandslideDataset as TheDataset

# your custom dataloader function
from dataset import get_dataloaders as the_dataloader

# hyperparameters from config
from config import batch_size as the_batch_size
from config import num_epochs as total_epochs
