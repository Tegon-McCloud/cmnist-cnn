import torch

from cmnist_cnn.model import MyAwesomeModel

def test_model_shape():
    model = MyAwesomeModel(lr=1e-3)
    input = torch.zeros(32, 1, 28, 28)
    output = model(input)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (32, 10)