import torch
import torch.utils.data.dataloader
import typer

from data import corrupt_mnist
from api import pick_device
from model import MyAwesomeModel


def evaluate(model_checkpoint: str, batch_size: int = 32) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    device = pick_device()

    print(f"{model_checkpoint=}, {batch_size=}, {device=}")

    model = MyAwesomeModel().to(device)
    model.load_state_dict(torch.load(model_checkpoint, weights_only=True))
    _, test_set = corrupt_mnist()

    test_loader = torch.utils.data.DataLoader(test_set, batch_size, shuffle=False)

    correct = 0
    total = 0

    for image, target in iter(test_loader):
        image, target = image.to(device), target.to(device)

        output = model(image)
        prediction = output.argmax(dim=1)

        correct += (prediction == target).sum().item()
        total += target.size(0)

    print(f"Test accuracy: {correct / total}")

if __name__ == "__main__":
    typer.run(evaluate)