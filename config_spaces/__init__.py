from . import cifar10
from . import cifar100
from . import dvc
from . import flower
from . import fmnist
from . import fruits
from . import mnist
from . import scmnist
from . import stl10
from . import svhn


DATASETS = [
    'cifar10',
    'cifar100',
    'fmnist',
    'mnist',
    'scmnist',
    'dvc',
    'svhn',
    'fruits',
    'stl10',
    'flower'
]


def get_config_space(dataset, seed):
    if dataset == 'cifar10':
        return cifar10.get_hyperparameter_search_space(seed)
    elif dataset == 'cifar100':
        return cifar100.get_hyperparameter_search_space(seed)
    elif dataset == 'dvc':
        return dvc.get_hyperparameter_search_space(seed)
    elif dataset == 'flower':
        return flower.get_hyperparameter_search_space(seed)
    elif dataset == 'fmnist':
        return fmnist.get_hyperparameter_search_space(seed)
    elif dataset == 'fruits':
        return fruits.get_hyperparameter_search_space(seed)
    elif dataset == 'mnist':
        return mnist.get_hyperparameter_search_space(seed)
    elif dataset == 'scmnist':
        return scmnist.get_hyperparameter_search_space(seed)
    elif dataset == 'stl10':
        return stl10.get_hyperparameter_search_space(seed)
    elif dataset == 'svhn':
        return svhn.get_hyperparameter_search_space(seed)
    else:
        raise ValueError('Unknown dataset: %s' % dataset)
