from . import cifar_scmnist_svhn  # covers cifar10, cifar100, svhn
from . import dvc
from . import flower
from . import fruits
from . import mnist  # covers mnist and fmnist
from . import stl10


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
        return cifar_scmnist_svhn.get_hyperparameter_search_space(seed)
    elif dataset == 'cifar100':
        return cifar_scmnist_svhn.get_hyperparameter_search_space(seed)
    elif dataset == 'dvc':
        return dvc.get_hyperparameter_search_space(seed)
    elif dataset == 'flower':
        return flower.get_hyperparameter_search_space(seed)
    elif dataset == 'fmnist':
        return mnist.get_hyperparameter_search_space(seed)
    elif dataset == 'fruits':
        return fruits.get_hyperparameter_search_space(seed)
    elif dataset == 'mnist':
        return mnist.get_hyperparameter_search_space(seed)
    elif dataset == 'scmnist':
        return cifar_scmnist_svhn.get_hyperparameter_search_space(seed)
    elif dataset == 'stl10':
        return stl10.get_hyperparameter_search_space(seed)
    elif dataset == 'svhn':
        return cifar_scmnist_svhn.get_hyperparameter_search_space(seed)
    else:
        raise ValueError('Unknown dataset: %s' % dataset)
