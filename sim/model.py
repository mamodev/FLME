import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First convolutional layer
        # Input channels (1 for grayscale MNIST), output channels (32), kernel size (3x3)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # Second convolutional layer
        # Input channels (32), output channels (64), kernel size (3x3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layer
        # After two conv layers and two pooling layers, the image size will be
        # (28 -> 14 -> 7). So, the flattened size will be 64 * 7 * 7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # Output layer (10 classes for digits 0-9)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Apply conv1, then ReLU, then pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Apply conv2, then ReLU, then pooling
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the output for the fully connected layers
        x = x.view(-1, 64 * 7 * 7)
        # Apply fc1, then ReLU
        x = F.relu(self.fc1(x))
        # Apply fc2 (output layer)
        x = self.fc2(x)
        return x


from torch.utils.data import TensorDataset, DataLoader

class GPULoader:
    def __init__(self, dataset, batch_size, shuffle=True, device=None, stream=None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.stream = stream if stream is not None else torch.cuda.current_stream(self.device)
        all_data, all_targets = [], []
        for img, label in dataset: # Direct iteration over dataset, slightly cleaner
            all_data.append(img)
            all_targets.append(label)
        with torch.cuda.stream(self.stream):
            self.data_tensor = torch.stack(all_data).to(self.device, non_blocking=True)
            self.target_tensor = torch.tensor(all_targets).to(self.device, non_blocking=True)
        self.gpu_dataset = TensorDataset(self.data_tensor, self.target_tensor)
        self.dataloader = DataLoader(self.gpu_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)
