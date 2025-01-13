from torch.utils.data import Dataset

from cmnist_cnn.data import corrupt_mnist

N_TRAIN = 30000
N_TEST = 5000



def test_corrupt_mnist():
    """Test the MyDataset class."""
    train_set, test_set = corrupt_mnist()
    
    for dataset, expected_size in [(train_set, N_TRAIN), (test_set, N_TEST)]:
        assert isinstance(dataset, Dataset)
        assert len(dataset) == expected_size
        assert all(x.shape == (1, 28, 28) for (x, _y) in dataset)
        assert all(0 <= y and y <= 9 for (_x, y) in dataset)


