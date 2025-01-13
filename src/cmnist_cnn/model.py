import torch
from torch import nn, optim
from pytorch_lightning import LightningModule

class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.resconv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.resconv = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        res = self.resconv(x)
        
        x = self.conv1(x)
        x = x.relu()
        x = self.conv2(x)

        return (x + res).relu()
    

class MyAwesomeModel(LightningModule):
    """My awesome model."""

    def __init__(self, lr: float) -> None:
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

        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        return self.classifier(features)
    
    def training_step(self, batch, batch_idx):
        images, target = batch
        output = self(images)
        loss = self.criterion(output, target)

        prediction = output.argmax(dim=1)
        correct = (prediction == target).sum().item()
        total = target.size(0)

        self.log("train_loss", loss.item(), prog_bar=True)
        self.log("train_acc", correct / total, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        images, target = batch
        output = self(images)
        loss = self.criterion(output, target)
        
        prediction = output.argmax(dim=1)
        correct = (prediction == target).sum().item()
        total = target.size(0)

        self.log("test_loss", loss.item(), on_epoch=True)
        self.log("test_acc", correct / total, on_epoch=True)

        return loss


    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

if __name__ == "__main__":

    model = MyAwesomeModel()
    param_count = sum(p.numel() for p in model.parameters())

    print(f"Model: {model}")
    print(f"Parameter count: {param_count}")
