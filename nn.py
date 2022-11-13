import torch
import util

class Conv(torch.nn.Module):
    def __init__(self,
                 dataset_type: util.Dataset):
        super().__init__()
        self._dataset_type = dataset_type

        self._encoder_mnist = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 32, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self._classifier_mnist = torch.nn.Sequential(
            torch.nn.Linear(512, 50),
            torch.nn.Tanh(),
            torch.nn.Linear(50, 10),
            torch.nn.Softmax(dim=1)
        )

        self._encoder_cifar = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 32, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )

        self._classifier_cifar = torch.nn.Sequential(
            torch.nn.Linear(800, 50),
            torch.nn.Tanh(),
            torch.nn.Linear(50, 10),
            torch.nn.Softmax(dim=1)
        )

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight, gain=1.414)

    def forward(self, x: torch.Tensor):
        if self._dataset_type == util.Dataset.MNIST:
            x = self._encoder_mnist(x)
            x = x.view(-1, 512)
            x = self._classifier_mnist(x)
        elif self._dataset_type == util.Dataset.CIFAR:
            x = self._encoder_cifar(x)
            x = x.view(-1, 800)
            x = self._classifier_cifar(x)
        else:
            x = None

        return x

class Attack(torch.nn.Module):
    def __init__(self,
                 input_dim: int=3):
        super().__init__()
        self._input_dim = input_dim

        self._mlp = torch.nn.Sequential(
            torch.nn.Linear(self._input_dim, 64),
            torch.nn.Linear(64, 2),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor):
        return self._mlp(x)
