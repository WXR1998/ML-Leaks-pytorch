import logging
import os
import enum
import pickle
import numpy as np

import configs
import dataset
import util


class Dataset(enum.Enum):
    CIFAR = 1
    MNIST = 2

class ModelType(enum.Enum):
    TARGET = 1
    SHADOW = 2
    ATTACK = 3

def get_dataset(data: Dataset):
    """
    :param data:
    :return:
    shadow_train, shadow_out, target_train, target_out, normalization
    """

    os.makedirs(configs.cache_path, exist_ok=True)
    filename = os.path.join(configs.cache_path, f'{data.name}.pkl')

    if os.path.exists(filename):
        logging.info('Loading dataset from cache...')
        with open(filename, 'rb') as fin:
            obj = pickle.load(fin)
        logging.info('Dataset loaded.')
    else:
        if data == Dataset.CIFAR:
            d, _, _ = dataset.read_cifar(configs.data_path)
        elif data == Dataset.MNIST:
            d, _, _ = dataset.read_mnist(configs.data_path)
        else:
            raise Exception('Unknown dataset.')

        x, _ = d
        mu = np.mean(x)
        sigma = np.std(x)
        normalization = (mu, sigma)

        shadow, target = dataset.split_half(d)
        shadow_train, shadow_out = dataset.split_half(shadow)
        target_train, target_out = dataset.split_half(target)

        obj = (shadow_train, shadow_out, target_train, target_out, normalization)

        with open(filename, 'wb') as fout:
            pickle.dump(obj, fout)
        logging.info('Dataset dumped.')

    return obj
