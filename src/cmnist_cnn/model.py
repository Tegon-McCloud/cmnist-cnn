from torch import nn

class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.resconv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.resconv = nn.Identity()

    def forward(self, x):

        res = self.resconv(x)
        
        x = self.conv1(x)
        x = x.relu()
        x = self.conv2(x)

        return (x + res).relu()



class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self) -> None:
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            ResBlock(16, 32),
            nn.MaxPool2d(2),
            ResBlock(32, 64),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64*3*3, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )


    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)
        

if __name__ == "__main__":

    model = MyAwesomeModel()
    param_count = sum(p.numel() for p in model.parameters())

    print(f"Model: {model}")
    print(f"Parameter count: {param_count}")
