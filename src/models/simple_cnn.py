from torch import nn

class CnnClassifier(nn.Module):

    def __init__(self, in_channels: int = 1, out_channels: int = 10):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, out_channels)
        )


    def forward(self, x):
        return self.model(x)