from torch import nn

TORCH_INPUT_SHAPE = (1, 3)
TORCH_OUTPUT_SHAPE = (1, 8)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = nn.Sequential(
            # nn.Linear(input_dim, 256),
            nn.Linear(TORCH_INPUT_SHAPE[1], 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, TORCH_OUTPUT_SHAPE[1]),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.fc(x)