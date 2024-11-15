import time
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

##########################
### MODEL DEFINITION
##########################

class LeNet5(torch.nn.Module):
    def __init__(self, num_classes=10, grayscale=True):
        super(LeNet5, self).__init__()
        in_channels = 1 if grayscale else 3
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 6, kernel_size=5),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(6, 16, kernel_size=5),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(16 * 5 * 5, 120),
            torch.nn.Tanh(),
            torch.nn.Linear(120, 84),
            torch.nn.Tanh(),
            torch.nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas

##########################
### LOADING THE TRAINED MODEL
##########################

# Initialize the model and load saved parameters
model = LeNet5()
model.load_state_dict(torch.load("lenet5_mnist.pth"))
model.eval()  # Set model to evaluation mode

# Use CPU for inference
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model.to(DEVICE)

##########################
### LOAD MNIST TEST DATASET
##########################

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = datasets.MNIST(
    root='./data', 
    train=False, 
    transform=transform, 
    download=False
)

test_loader = DataLoader(
    dataset=test_dataset, 
    batch_size=1000,  # Larger batch size for faster inference timing
    shuffle=False
)

##########################
### INFERENCE FUNCTION
##########################

def run_inference(model, data_loader, device):
    correct = 0
    total = 0
    start_time = time.time()

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            logits, probas = model(images)
            _, predicted = torch.max(probas, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    end_time = time.time()
    accuracy = 100 * correct / total  # Calculate accuracy
    inference_time = end_time - start_time  # Total inference time
    throughput = total / inference_time  # Throughput: images per second

    print(f'Accuracy: {accuracy:.2f}%')
    print(f'Inference Time: {inference_time:.4f} seconds')
    print(f'Throughput: {throughput:.2f} images per second')

##########################
### RUN INFERENCE AND BASELINE MEASUREMENT
##########################

run_inference(model, test_loader, DEVICE)


