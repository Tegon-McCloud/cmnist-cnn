from torch.utils.data import Dataset
import pytest

import os.path

from cmnist_cnn.data import corrupt_mnist

N_TRAIN = 30000
N_TEST = 5000


@pytest.mark.skipif(not os.path.exists("data/"), reason="Data directory missing")
def test_corrupt_mnist():
    """Test the MyDataset class."""
    train_set, test_set = corrupt_mnist()
    
    for dataset, expected_size, name in [(train_set, N_TRAIN, "train"), (test_set, N_TEST, "test")]:
        assert isinstance(dataset, Dataset), f"{name} set is not an instance of Dataset"
        assert len(dataset) == expected_size, f"{name} set does not have the expected length (got {len(dataset)}, expected {expected_size})"
        assert all(x.shape == (1, 28, 28) for (x, _y) in dataset), f"{name} set contains images not of shape (1, 28, 28)"
        assert all(0 <= y and y <= 9 for (_x, y) in dataset), f"{name} set contains targets that are not between 0 and 9"


