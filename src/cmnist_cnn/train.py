import torch
import torch.utils.data.dataloader
import typer
import torch.nn as nn
import matplotlib.pyplot as plt

import os

from data import corrupt_mnist
from utils import pick_device
from model import MyAwesomeModel

app = typer.Typer()

def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 10) -> None:
    """Train a model on MNIST."""

    device = pick_device()

    print("Training day and night")
    print(f"{lr=}, {batch_size=}, {epochs=}, {device=}")

    model = MyAwesomeModel().to(device)
    train_set, _ = corrupt_mnist()

    train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr)
    criterion = nn.CrossEntropyLoss()

    loss_history = []
    acc_history = []
    
    model.train()
    for epoch in range(epochs):
        for image, target in iter(train_loader):
            image, target = image.to(device), target.to(device)

            output = model(image)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prediction = output.argmax(dim=1)
            correct = (prediction == target).sum().item()
            total = target.size(0)

            loss_history.append(loss.item())
            acc_history.append(correct / total)
    
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/model.pt")

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    axs[0].plot(loss_history)
    axs[0].set_title("Training loss")
    axs[0].set_xlabel("Step")
    
    axs[1].plot(acc_history)
    axs[1].set_title("Training Accuracy")
    axs[1].set_xlabel("Step")
    
    fig.tight_layout()
    
    os.makedirs("reports/figures", exist_ok=True)
    fig.savefig("reports/figures/train_stats.pdf")


if __name__ == "__main__":
    typer.run(train)