import torch
import torch.utils.data.dataloader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning.loggers
import typer
import torch.nn as nn
import matplotlib.pyplot as plt

import os

from data import corrupt_mnist
from utils import pick_device
from model import MyAwesomeModel

app = typer.Typer()

def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 10) -> None:
    device = pick_device()
    model = MyAwesomeModel(lr).to(device)

    train_set, _ = corrupt_mnist()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, num_workers=4, persistent_workers=True)

    checkpoint_callback = ModelCheckpoint(dirpath="models", filename="{epoch}")

    trainer = Trainer(
        max_epochs=epochs,
        callbacks=[checkpoint_callback],
        logger=pytorch_lightning.loggers.WandbLogger(project="cmnist-cnn"),
    )
    trainer.fit(model, train_loader)


if __name__ == "__main__":
    typer.run(train)