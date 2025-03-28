
import torch
import os
import io

from torch import nn, optim
from torchvision import transforms
import torchvision

def __diff(state_dict, reference_dict):
    """Compute model update differences."""
    diff = {key: state_dict[key] - reference_dict[key] for key in state_dict}
    return diff


def __dict_to_buffer(state_dict):
    """Convert a state_dict to a buffer."""
    buffer = io.BytesIO()
    for key in state_dict:
        buffer.write(memoryview(state_dict[key].untyped_storage()))

    return buffer

def encode(state_dict, reference_dict):
    """Main function: Compute difference, quantize, sparsify, and compress."""
    diff = __diff(state_dict, reference_dict)


class SimpleMnistNN(torch.nn.Module):
    def __init__(self):
        super(SimpleMnistNN, self).__init__()
        self.fc1 = torch.nn.Linear(784, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def load_data(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def train(model, train_loader, criterion, optimizer, epochs=5, device="cpu"):
    model.to(device)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = SimpleMnistNN()
    train_loader, test_loader = load_data()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    os.makedirs("./models", exist_ok=True)

    for i in range(10):
        train(model, train_loader, criterion, optimizer, epochs=5, device=device)
        # save model to folder ./models/round_{i}
        torch.save(model.state_dict(), f"./models/round_{i}/model.pth")

