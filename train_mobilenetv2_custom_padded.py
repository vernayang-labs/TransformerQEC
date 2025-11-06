import yaml, os, numpy as np
from torch.utils.data import DataLoader, random_split

import lightning as L
from lightning.pytorch import seed_everything

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from tqdm import tqdm

from data import QuantumSyndromeDataset

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# Load configuration from YAML file
project_root=os.getcwd()
config_file = os.path.join(project_root, 'config.yaml')
with open(config_file, "r") as file:
    data = yaml.safe_load(file)

# To get reproducibility
seed = data["SEED"]
if seed:
    seed_everything(42, workers=True)

# Extract configuration parameters
BATCH_SIZE = data["BATCH_SIZE"]
DATASET_DIR = data["DATASET_DIR"]

# Read data from the last generated CSV file using polars
index = len(os.listdir(DATASET_DIR))
datafile = os.path.join(project_root, DATASET_DIR, f"data{index-1}.csv")

# dataset and dataloader
dataset = QuantumSyndromeDataset(datafile)
l = len(dataset) * np.array(
    (data["TRAIN_SPLIT"], data["VALID_SPLIT"], data["TEST_SPLIT"])
)
train_ds, val_ds, test_ds = random_split(dataset, l.astype(int))


# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, data, normalize=True):  # Pass the full list of tuples
        self.inputs = [item[0] for item in data]  # Extract all input tensors
        self.outputs = [item[1] for item in data]  # Extract all output tensors
        self.normalize = normalize

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inp = self.inputs[idx].unsqueeze(0)  # [1, 24, 5]
        # Pad to [1, 32, 32]
        pad_left = 13
        pad_right = 14  # 5 + 13 + 14 = 32
        pad_top = 4
        pad_bottom = 4  # 24 + 4 + 4 = 32
        inp = nn.functional.pad(inp, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        
        if self.normalize:
            inp = (inp - 0.485) / 0.229  # Approximate single-channel norm
        
        out = self.outputs[idx]
        return inp, out
    

train_dataset = CustomDataset(train_ds)
test_dataset = CustomDataset(test_ds)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Load pretrained MobileNetV2 and customize
model = models.mobilenet_v2(pretrained=True)

# Customize first conv for 1 channel: average pretrained weights over channels
first_conv = model.features[0][0]
new_first_conv = nn.Conv2d(1, first_conv.out_channels, kernel_size=first_conv.kernel_size,
                           stride=first_conv.stride, padding=first_conv.padding, bias=first_conv.bias)
with torch.no_grad():
    new_first_conv.weight.copy_(first_conv.weight.mean(dim=1, keepdim=True))  # Average RGB to grayscale
model.features[0][0] = new_first_conv

# Modify classifier for 363 binary outputs
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 363)

model = model.to(device)

# Loss (multi-label binary) and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train(epoch):
    model.train()
    running_loss = 0.0
    total_acc = 0.0
    total_samples = 0
    for data in tqdm(train_loader, desc=f'Epoch {epoch} Training'):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        acc = (preds == labels).float().mean()
        total_acc += acc * labels.size(0)
        total_samples += labels.size(0)
    
    print(f'Epoch {epoch} - Loss: {running_loss / len(train_loader):.4f}, Accuracy: {total_acc / total_samples:.4f}')

# Validation function
def validate():
    model.eval()
    running_loss = 0.0
    total_acc = 0.0
    total_samples = 0
    with torch.no_grad():
        for data in tqdm(test_loader, desc='Validating'):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            acc = (preds == labels).float().mean()
            total_acc += acc * labels.size(0)
            total_samples += labels.size(0)
    
    print(f'Validation - Loss: {running_loss / len(test_loader):.4f}, Accuracy: {total_acc / total_samples:.4f}')

# Train for 5 epochs (adjust as needed)
num_epochs = 5
for epoch in range(1, num_epochs + 1):
    train(epoch)
    validate()

# Save the trained model
torch.save(model.state_dict(), 'mobilenetv2_custom_padded.pth')
print('Model saved as mobilenetv2_custom_padded.pth')