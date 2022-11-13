import logging
import os
import pickle
from typing import Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision

import configs

def read_mnist(path: str) -> tuple:
    logging.info('Start reading mnist...')

    mnist_train = torchvision.datasets.MNIST(path, download=True, train=True)
    mnist_test = torchvision.datasets.MNIST(path, download=True, train=False)

    train_x = mnist_train.data.detach().numpy() / 256
    train_x = np.expand_dims(train_x, 1)
    train_y = mnist_train.targets.detach().numpy()

    test_x = mnist_test.data.detach().numpy() / 256
    test_x = np.expand_dims(test_x, 1)
    test_y = mnist_test.targets.detach().numpy()

    logging.info('Finish reading mnist.')
    return (train_x, train_y), (test_x, test_y), mnist_train.classes

def read_cifar(path: str) -> tuple:
    logging.info('Start reading cifar...')

    cifar_train = torchvision.datasets.CIFAR10(path, download=True, train=True)
    cifar_test = torchvision.datasets.CIFAR10(path, download=True, train=False)

    train_x = cifar_train.data / 256
    train_y = np.array(cifar_train.targets)
    train_x = train_x.transpose(0, 3, 1, 2)

    test_x = cifar_test.data / 256
    test_y = np.array(cifar_test.targets)
    test_x = test_x.transpose(0, 3, 1, 2)

    logging.info('Finish reading cifar...')
    return (train_x, train_y), (test_x, test_y), cifar_train.classes

def split_half(d: tuple) -> tuple:
    x, y = d
    assert len(x) == len(y)
    perm = np.random.permutation(len(x))
    x_ = x[perm]
    y_ = y[perm]

    l = len(x) // 2

    return (x_[:l], y_[:l]), (x_[l:], y_[l:])

def feature_vector(p: np.ndarray) -> np.ndarray:
    c = p.shape[1]
    p = np.sort(p, axis=1)
    return p[:, -min(3, c):]

def dump_D(positive: np.ndarray, negative: np.ndarray, filename: str):
    os.makedirs(configs.ckpt_path, exist_ok=True)
    save_obj = {
        'positive': positive,
        'negative': negative
    }
    path = os.path.join(configs.ckpt_path, f'{filename}_D.pkl')
    with open(path, 'wb') as fout:
        pickle.dump(save_obj, fout)

def load_D(filename: str) -> tuple:
    path = os.path.join(configs.ckpt_path, f'{filename}_D.pkl')
    with open(path, 'rb') as fin:
        obj = pickle.load(fin)
    return obj['positive'], obj['negative']

def get_dataloader(
        d: tuple,
        batch_size: int=configs.batch_size,
        normalize: Optional[tuple]=None,
        shuffle: bool=False,
):
    x, y = d
    if normalize is not None:
        mu, sigma = normalize
        x = (np.array(x) - mu) / sigma

    tensor_dataset = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long)
    )
    loader = DataLoader(
        tensor_dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    return loader
