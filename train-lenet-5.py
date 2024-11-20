import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

##########################
### SETTINGS
##########################

# Hyperparameters
RANDOM_SEED = 1
LEARNING_RATE = 0.001
BATCH_SIZE = 128
NUM_EPOCHS = 5  # Reduced for faster testing, increase if needed

# Architecture
NUM_FEATURES = 32 * 32
NUM_CLASSES = 10

# Other
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GRAYSCALE = True

##########################
### MNIST DATASET
##########################

resize_transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize to 32x32 for LeNet-5
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to match training setup
])

# Load MNIST dataset
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    transform=resize_transform,
    download=False  # Assuming dataset is already downloaded
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    transform=resize_transform,
    download=False
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

##########################
### MODEL
##########################

class LeNet5(nn.Module):

    def __init__(self, num_classes, grayscale=False):
        super(LeNet5, self).__init__()
        self.grayscale = grayscale
        self.num_classes = num_classes

        # Input channels: 1 for grayscale, 3 for RGB
        in_channels = 1 if self.grayscale else 3

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas


##########################
### TRAINING LOOP
##########################

torch.manual_seed(RANDOM_SEED)

# Initialize the model
model = LeNet5(NUM_CLASSES, GRAYSCALE)
model.to(DEVICE)
#model = torch.jit.script(model)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Function to compute accuracy
def compute_accuracy(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # No gradient calculation for evaluation
        for features, targets in data_loader:
            features, targets = features.to(device), targets.to(device)
            logits, probas = model(features)
            _, predicted_labels = torch.max(probas, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float() / num_examples * 100


# Training loop
start_time = time.time()
for epoch in range(NUM_EPOCHS):

    model.train()  # Set model to training mode
    for batch_idx, (features, targets) in enumerate(train_loader):
        features, targets = features.to(DEVICE), targets.to(DEVICE)

        # Forward and backward pass
        logits, probas = model(features)
        cost = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # Logging
        if batch_idx % 50 == 0:
            print(f'Epoch: {epoch + 1:03}/{NUM_EPOCHS:03} | '
                  f'Batch {batch_idx:04}/{len(train_loader)} | Cost: {cost:.4f}')

    # Compute accuracy for this epoch
    train_acc = compute_accuracy(model, train_loader, DEVICE)
    print(f'Epoch: {epoch + 1:03}/{NUM_EPOCHS:03} | Train Accuracy: {train_acc:.2f}%')

    print(f'Time elapsed: {((time.time() - start_time) / 60):.2f} min')

print(f'Total Training Time: {((time.time() - start_time) / 60):.2f} min')

##########################
### SAVE THE MODEL
##########################

torch.save(model.state_dict(), "lenet5_mnist.pth")
print("Model saved as lenet5_mnist.pth")

##########################
### TEST THE MODEL
##########################

# Load the model for inference
model.load_state_dict(torch.load("lenet5_mnist.pth"))
model.eval()

# Compute test accuracy
test_acc = compute_accuracy(model, test_loader, DEVICE)
print(f'Test Accuracy: {test_acc:.2f}%')
