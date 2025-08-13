from torch import nn
from torch.functional import F

class SimpleModel(nn.Module):
    def __init__(self, insize=2, outsize=8):
        super(SimpleModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(insize, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, outsize),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.fc(x)

class LogisticRegression(nn.Module):
    def __init__(self, insize=2, outsize=8):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(insize, outsize)

    def forward(self, x):
        logits = self.linear(x)
        return F.log_softmax(logits, dim=1)
    
class DeeperModel(nn.Module):
    def __init__(self, insize=2, outsize=8):
        super(DeeperModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(insize, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),  # Added layer
            nn.ReLU(),              # Added layer
            nn.Linear(256, 128),  # Added layer
            nn.ReLU(),              # Added layer
            nn.Linear(128, outsize),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.fc(x)
    
class VeryDeepModel(nn.Module):
    def __init__(self, insize=2, outsize=8):
        super(VeryDeepModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(insize, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, outsize),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.fc(x)


MODELS = {
    "simple": SimpleModel,
    "logistic": LogisticRegression,
    "deeper": DeeperModel,
    "very_deep": VeryDeepModel
}