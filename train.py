"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import data
import engine
import VIT_model
import utils
import pathlib
from torchvision import transforms

# Setup hyperparameters
NUM_EPOCHS = 5
LEARNING_RATE = 0.0001  # Reduced learning rate for stability
data_path = pathlib.Path("data/")
train_dl, test_dl, class_names = data.data_setup(
    source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
    destination="pizza_steak_sushi"
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Create model with help from model_builder.py
model = VIT_model.ViT(num_classes=3)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# Start training with help from engine.py
result = engine.train(model=model,
                      train_dataloader=train_dl,
                      test_dataloader=test_dl,
                      loss_fn=loss_fn,
                      optimizer=optimizer,
                      epochs=NUM_EPOCHS,
                      device=device)

# Save the model with help from utils.py
utils.save_model(model=model,
                 target_dir="models",
                 model_name="VIT_model.pth")
