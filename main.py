# Import necessary libraries
import os

import torch
import torch.nn as nn
import torch.optim as optim
import json


# Define the neural network architecture
from torch.utils.data import DataLoader, random_split

from loaddata import WeatherDataset
from model import WeatherTimeModel
from train import Trainer


# Set the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the dataset
train_dataset = WeatherDataset('data/train_dataset')
train_dataset.show_weatherAndperiod_count()
# Define the size of the validation set
val_size = 300

# Calculate the size of the training set
train_size = len(train_dataset) - val_size

# Save the model
if os.path.exists('./saved_models_18') is False:
    os.mkdir('./saved_models_18')
model_save_path = './saved_models_18/model.pth'

# Use the random_split function to split the dataset into training and validation sets
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# Define the data loader for the training set
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define the data loader for the validation set
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Create the model, loss function, and optimizer
model = WeatherTimeModel()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
criterion = nn.CrossEntropyLoss()
criterion.to(device)

trainer = Trainer(model, train_loader, val_loader, val_dataset, criterion, optimizer, device, model_save_path)
trainer.train()
print('Training complete!')

