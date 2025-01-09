import torch
import torch.nn as nn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pathlib import Path

from api import pick_device
from model import MyAwesomeModel
from data import corrupt_mnist


def load_model(model_checkpoint: Path) -> MyAwesomeModel:
    model = MyAwesomeModel()
    model.load_state_dict(torch.load(model_checkpoint, weights_only=True))
    model.eval()

    return model


class InputStore:
    def __init__(self) -> None:
        self.values = []

    def __call__(self, _module: nn.Module, input: torch.Tensor, _output: torch.Tensor):
        self.values.append(input[0].detach())


if __name__ == "__main__":
    device = pick_device()
    model = load_model("models/model.pt").to(device)

    _, test_set = corrupt_mnist()
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

    input_store = InputStore()
    hook_handle = model.classifier[-2].register_forward_hook(input_store)  # index -2 is last ReLU

    for image, target in iter(test_loader):
        image, target = image.to(device), target.to(device)
        model(image)

    hook_handle.remove()
    features = torch.cat(input_store.values)

    embeddings = TSNE(n_components=2, learning_rate="auto", init="random", perplexity=3).fit_transform(features)

    fig, ax = plt.subplots(1, 1)

    ax.scatter(embeddings[:, 0], embeddings[:, 1])

    fig.savefig("reports/figures/tsne.pdf")
