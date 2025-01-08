import torch
import typer
from pathlib import Path


def normalize_data(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean()) / x.std()


def preprocess_data(raw_data_dir: Path, output_dir: Path):
    """Load data from raw_data_dir, process it and save the results to output_dir"""
    train_images = torch.concat([torch.load(raw_data_dir / f"train_images_{i}.pt", weights_only=True) for i in range(6)])
    train_target = torch.concat([torch.load(raw_data_dir / f"train_target_{i}.pt", weights_only=True) for i in range(6)])

    test_images = torch.load(raw_data_dir / "test_images.pt", weights_only=True)
    test_target = torch.load(raw_data_dir / "test_target.pt", weights_only=True)
    
    train_images = normalize_data(train_images)[:,None,:,:]
    test_images = normalize_data(test_images)[:,None,:,:]

    torch.save(train_images, output_dir / "train_images.pt")
    torch.save(train_target, output_dir / "train_target.pt")
    torch.save(test_images, output_dir / "test_images.pt")
    torch.save(test_target, output_dir / "test_target.pt")
    


def corrupt_mnist(processed_data_dir: Path = Path("data/processed")):
    """Return train and test dataloaders for corrupt MNIST."""

    train_images = torch.load(processed_data_dir / "train_images.pt", weights_only=True)
    train_target = torch.load(processed_data_dir / "train_target.pt", weights_only=True)
    test_images = torch.load(processed_data_dir / "test_images.pt", weights_only=True)
    test_target = torch.load(processed_data_dir / "test_target.pt", weights_only=True)

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)

    return train_set, test_set


if __name__ == "__main__":
    typer.run(preprocess_data)