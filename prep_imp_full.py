import numpy as np
from fanova import fANOVA
import ConfigSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
import sys
import os
import csv
path = os.path.dirname(os.path.realpath(__file__))



def get_hyperparameter_search_space_cifar10(seed=None):
    """
    Neural Network search space based on a best effort using the scikit-learn
    implementation. Note that for state of the art performance, other packages
    could be preferred.

    Parameters
    ----------
    seed: int
        Random seed that will be used to sample random configurations

    Returns
    -------
    cs: ConfigSpace.ConfigurationSpace
        The configuration space object
    """
    cs = ConfigSpace.ConfigurationSpace('ResNet18_classifier', seed)
    learning_rate_init = ConfigSpace.UniformFloatHyperparameter(
        name='learning_rate_init', lower=1e-6, upper=1, log=True, default_value=1e-1)
    epochs = ConfigSpace.UniformIntegerHyperparameter(
        name='epochs', lower=1, upper=200, default_value=150)
    batch_size = ConfigSpace.CategoricalHyperparameter(
        name='batch_size', choices=[32, 64, 128, 256, 512], default_value=128)
    momentum = ConfigSpace.UniformFloatHyperparameter(
        name='momentum', lower=0, upper=1, default_value=0.9)
    weight_decay = ConfigSpace.UniformFloatHyperparameter(
        name='weight_decay', lower=1e-6, upper=1e-2, log=True, default_value=5e-4)
    lr_decay = ConfigSpace.UniformIntegerHyperparameter(
        name='lr_decay', lower=2, upper=1000, log=True, default_value=10)
    patience = ConfigSpace.UniformIntegerHyperparameter(
        name='patience', lower=2, upper=200, log=False, default_value=10)
    tolerance = ConfigSpace.UniformFloatHyperparameter(
        name='tolerance', lower=1e-5, upper=1e-2, log=True, default_value=1e-4)
    resize_crop = ConfigSpace.CategoricalHyperparameter(
        name='resize_crop', choices=[True, False], default_value=False)
    h_flip = ConfigSpace.CategoricalHyperparameter(
        name='h_flip', choices=[True, False], default_value=False)
    v_flip = ConfigSpace.CategoricalHyperparameter(
        name='v_flip', choices=[True, False], default_value=False)
    shuffle = ConfigSpace.CategoricalHyperparameter(
        name='shuffle', choices=[True, False], default_value=True)

    cs.add_hyperparameters([
        batch_size,
        learning_rate_init,
        epochs,
        momentum,
        weight_decay,
        lr_decay,
        patience,
        tolerance,
        resize_crop,
        h_flip,
        v_flip,
        shuffle,
    ])

    return cs

def get_hyperparameter_search_space_cifar100(seed=None):
    """
    Neural Network search space based on a best effort using the scikit-learn
    implementation. Note that for state of the art performance, other packages
    could be preferred.

    Parameters
    ----------
    seed: int
        Random seed that will be used to sample random configurations

    Returns
    -------
    cs: ConfigSpace.ConfigurationSpace
        The configuration space object
    """
    cs = ConfigSpace.ConfigurationSpace('ResNet18_classifier', seed)
    learning_rate_init = ConfigSpace.UniformFloatHyperparameter(
        name='learning_rate_init', lower=1e-6, upper=1, log=True, default_value=1e-1)
    epochs = ConfigSpace.UniformIntegerHyperparameter(
        name='epochs', lower=1, upper=200, default_value=150)
    batch_size = ConfigSpace.CategoricalHyperparameter(
        name='batch_size', choices=[32, 64, 128, 256, 512], default_value=128)
    momentum = ConfigSpace.UniformFloatHyperparameter(
        name='momentum', lower=0, upper=1, default_value=0.9)
    weight_decay = ConfigSpace.UniformFloatHyperparameter(
        name='weight_decay', lower=1e-6, upper=1e-2, log=True, default_value=5e-4)
    lr_decay = ConfigSpace.UniformIntegerHyperparameter(
        name='lr_decay', lower=2, upper=1000, log=True, default_value=10)
    patience = ConfigSpace.UniformIntegerHyperparameter(
        name='patience', lower=2, upper=200, log=False, default_value=10)
    tolerance = ConfigSpace.UniformFloatHyperparameter(
        name='tolerance', lower=1e-5, upper=1e-2, log=True, default_value=1e-4)
    resize_crop = ConfigSpace.CategoricalHyperparameter(
        name='resize_crop', choices=[True, False], default_value=False)
    h_flip = ConfigSpace.CategoricalHyperparameter(
        name='h_flip', choices=[True, False], default_value=False)
    v_flip = ConfigSpace.CategoricalHyperparameter(
        name='v_flip', choices=[True, False], default_value=False)
    shuffle = ConfigSpace.CategoricalHyperparameter(
        name='shuffle', choices=[True, False], default_value=True)

    cs.add_hyperparameters([
        batch_size,
        learning_rate_init,
        epochs,
        momentum,
        weight_decay,
        lr_decay,
        patience,
        tolerance,
        resize_crop,
        h_flip,
        v_flip,
        shuffle,
    ])

    return cs

def get_hyperparameter_search_space_mnist(seed=None):
    """
    Neural Network search space based on a best effort using the scikit-learn
    implementation. Note that for state of the art performance, other packages
    could be preferred.

    Parameters
    ----------
    seed: int
        Random seed that will be used to sample random configurations

    Returns
    -------
    cs: ConfigSpace.ConfigurationSpace
        The configuration space object
    """
    cs = ConfigSpace.ConfigurationSpace('ResNet18_classifier', seed)
    learning_rate_init = ConfigSpace.UniformFloatHyperparameter(
        name='learning_rate_init', lower=1e-6, upper=1e-1, log=True, default_value=1e-2)
    epochs = ConfigSpace.UniformIntegerHyperparameter(
        name='epochs', lower=1, upper=50, default_value=20)
    batch_size = ConfigSpace.CategoricalHyperparameter(
        name='batch_size', choices=[32, 64, 128, 256, 512], default_value=128)
    momentum = ConfigSpace.UniformFloatHyperparameter(
        name='momentum', lower=0, upper=1, default_value=0.9)
    weight_decay = ConfigSpace.UniformFloatHyperparameter(
        name='weight_decay', lower=1e-6, upper=1e-2, log=True, default_value=5e-4)
    lr_decay = ConfigSpace.UniformIntegerHyperparameter(
        name='lr_decay', lower=2, upper=1000, log=True, default_value=10)
    patience = ConfigSpace.UniformIntegerHyperparameter(
        name='patience', lower=2, upper=200, log=False, default_value=10)
    tolerance = ConfigSpace.UniformFloatHyperparameter(
        name='tolerance', lower=1e-5, upper=1e-2, log=True, default_value=1e-4)
    resize_crop = ConfigSpace.CategoricalHyperparameter(
        name='resize_crop', choices=[True, False], default_value=False)
    h_flip = ConfigSpace.CategoricalHyperparameter(
        name='h_flip', choices=[True, False], default_value=False)
    v_flip = ConfigSpace.CategoricalHyperparameter(
        name='v_flip', choices=[True, False], default_value=False)
    shuffle = ConfigSpace.CategoricalHyperparameter(
        name='shuffle', choices=[True, False], default_value=True)

    cs.add_hyperparameters([
        batch_size,
        learning_rate_init,
        epochs,
        momentum,
        weight_decay,
        lr_decay,
        patience,
        tolerance,
        resize_crop,
        h_flip,
        v_flip,
        shuffle,
    ])

    return cs

def get_hyperparameter_search_space_fmnist(seed=None):
    """
    Neural Network search space based on a best effort using the scikit-learn
    implementation. Note that for state of the art performance, other packages
    could be preferred.

    Parameters
    ----------
    seed: int
        Random seed that will be used to sample random configurations

    Returns
    -------
    cs: ConfigSpace.ConfigurationSpace
        The configuration space object
    """
    cs = ConfigSpace.ConfigurationSpace('ResNet18_classifier', seed)
    learning_rate_init = ConfigSpace.UniformFloatHyperparameter(
        name='learning_rate_init', lower=1e-6, upper=1e-1, log=True, default_value=1e-2)
    epochs = ConfigSpace.UniformIntegerHyperparameter(
        name='epochs', lower=1, upper=50, default_value=20)
    batch_size = ConfigSpace.CategoricalHyperparameter(
        name='batch_size', choices=[32, 64, 128, 256, 512], default_value=128)
    momentum = ConfigSpace.UniformFloatHyperparameter(
        name='momentum', lower=0, upper=1, default_value=0.9)
    weight_decay = ConfigSpace.UniformFloatHyperparameter(
        name='weight_decay', lower=1e-6, upper=1e-2, log=True, default_value=5e-4)
    lr_decay = ConfigSpace.UniformIntegerHyperparameter(
        name='lr_decay', lower=2, upper=1000, log=True, default_value=10)
    patience = ConfigSpace.UniformIntegerHyperparameter(
        name='patience', lower=2, upper=200, log=False, default_value=10)
    tolerance = ConfigSpace.UniformFloatHyperparameter(
        name='tolerance', lower=1e-5, upper=1e-2, log=True, default_value=1e-4)
    resize_crop = ConfigSpace.CategoricalHyperparameter(
        name='resize_crop', choices=[True, False], default_value=False)
    h_flip = ConfigSpace.CategoricalHyperparameter(
        name='h_flip', choices=[True, False], default_value=False)
    v_flip = ConfigSpace.CategoricalHyperparameter(
        name='v_flip', choices=[True, False], default_value=False)
    shuffle = ConfigSpace.CategoricalHyperparameter(
        name='shuffle', choices=[True, False], default_value=True)

    cs.add_hyperparameters([
        batch_size,
        learning_rate_init,
        epochs,
        momentum,
        weight_decay,
        lr_decay,
        patience,
        tolerance,
        resize_crop,
        h_flip,
        v_flip,
        shuffle,
    ])

    return cs

def get_hyperparameter_search_space_scmnist(seed=None):
    """
    Neural Network search space based on a best effort using the scikit-learn
    implementation. Note that for state of the art performance, other packages
    could be preferred.

    Parameters
    ----------
    seed: int
        Random seed that will be used to sample random configurations

    Returns
    -------
    cs: ConfigSpace.ConfigurationSpace
        The configuration space object
    """
    cs = ConfigSpace.ConfigurationSpace('ResNet18_classifier', seed)
    learning_rate_init = ConfigSpace.UniformFloatHyperparameter(
        name='learning_rate_init', lower=1e-6, upper=1, log=True, default_value=1e-1)
    epochs = ConfigSpace.UniformIntegerHyperparameter(
        name='epochs', lower=1, upper=200, default_value=150)
    batch_size = ConfigSpace.CategoricalHyperparameter(
        name='batch_size', choices=[32, 64, 128, 256, 512], default_value=128)
    momentum = ConfigSpace.UniformFloatHyperparameter(
        name='momentum', lower=0, upper=1, default_value=0.9)
    weight_decay = ConfigSpace.UniformFloatHyperparameter(
        name='weight_decay', lower=1e-6, upper=1e-2, log=True, default_value=5e-4)
    lr_decay = ConfigSpace.UniformIntegerHyperparameter(
        name='lr_decay', lower=2, upper=1000, log=True, default_value=10)
    patience = ConfigSpace.UniformIntegerHyperparameter(
        name='patience', lower=2, upper=200, log=False, default_value=10)
    tolerance = ConfigSpace.UniformFloatHyperparameter(
        name='tolerance', lower=1e-5, upper=1e-2, log=True, default_value=1e-4)
    resize_crop = ConfigSpace.CategoricalHyperparameter(
        name='resize_crop', choices=[True, False], default_value=False)
    h_flip = ConfigSpace.CategoricalHyperparameter(
        name='h_flip', choices=[True, False], default_value=False)
    v_flip = ConfigSpace.CategoricalHyperparameter(
        name='v_flip', choices=[True, False], default_value=False)
    shuffle = ConfigSpace.CategoricalHyperparameter(
        name='shuffle', choices=[True, False], default_value=True)

    cs.add_hyperparameters([
        batch_size,
        learning_rate_init,
        epochs,
        momentum,
        weight_decay,
        lr_decay,
        patience,
        tolerance,
        resize_crop,
        h_flip,
        v_flip,
        shuffle,
    ])

    return cs

def get_hyperparameter_search_space_dvc(seed=None):
    """
    Neural Network search space based on a best effort using the scikit-learn
    implementation. Note that for state of the art performance, other packages
    could be preferred.

    Parameters
    ----------
    seed: int
        Random seed that will be used to sample random configurations

    Returns
    -------
    cs: ConfigSpace.ConfigurationSpace
        The configuration space object
    """
    cs = ConfigSpace.ConfigurationSpace('ResNet18_classifier', seed)
    learning_rate_init = ConfigSpace.UniformFloatHyperparameter(
        name='learning_rate_init', lower=1e-6, upper=1e-1, log=True, default_value=1e-1)
    epochs = ConfigSpace.UniformIntegerHyperparameter(
        name='epochs', lower=1, upper=100, default_value=80)
    batch_size = ConfigSpace.CategoricalHyperparameter(
        name='batch_size', choices=[8, 16, 32, 64, 128], default_value=128)
    momentum = ConfigSpace.UniformFloatHyperparameter(
        name='momentum', lower=0, upper=1, default_value=0.9)
    weight_decay = ConfigSpace.UniformFloatHyperparameter(
        name='weight_decay', lower=1e-6, upper=1e-2, log=True, default_value=5e-4)
    lr_decay = ConfigSpace.UniformIntegerHyperparameter(
        name='lr_decay', lower=2, upper=1000, log=True, default_value=10)
    patience = ConfigSpace.UniformIntegerHyperparameter(
        name='patience', lower=2, upper=200, log=False, default_value=10)
    tolerance = ConfigSpace.UniformFloatHyperparameter(
        name='tolerance', lower=1e-5, upper=1e-2, log=True, default_value=1e-4)
    resize_crop = ConfigSpace.CategoricalHyperparameter(
        name='resize_crop', choices=[True, False], default_value=False)
    h_flip = ConfigSpace.CategoricalHyperparameter(
        name='h_flip', choices=[True, False], default_value=False)
    v_flip = ConfigSpace.CategoricalHyperparameter(
        name='v_flip', choices=[True, False], default_value=False)
    shuffle = ConfigSpace.CategoricalHyperparameter(
        name='shuffle', choices=[True, False], default_value=True)

    cs.add_hyperparameters([
        batch_size,
        learning_rate_init,
        epochs,
        momentum,
        weight_decay,
        lr_decay,
        patience,
        tolerance,
        resize_crop,
        h_flip,
        v_flip,
        shuffle,
    ])

    return cs

def get_hyperparameter_search_space_svhn(seed=None):
    """
    Neural Network search space based on a best effort using the scikit-learn
    implementation. Note that for state of the art performance, other packages
    could be preferred.

    Parameters
    ----------
    seed: int
        Random seed that will be used to sample random configurations

    Returns
    -------
    cs: ConfigSpace.ConfigurationSpace
        The configuration space object
    """
    cs = ConfigSpace.ConfigurationSpace('ResNet18_classifier', seed)
    learning_rate_init = ConfigSpace.UniformFloatHyperparameter(
        name='learning_rate_init', lower=1e-6, upper=1, log=True, default_value=1e-1)
    epochs = ConfigSpace.UniformIntegerHyperparameter(
        name='epochs', lower=1, upper=200, default_value=100)
    batch_size = ConfigSpace.CategoricalHyperparameter(
        name='batch_size', choices=[32, 64, 128, 256, 512], default_value=128)
    momentum = ConfigSpace.UniformFloatHyperparameter(
        name='momentum', lower=0, upper=1, default_value=0.9)
    weight_decay = ConfigSpace.UniformFloatHyperparameter(
        name='weight_decay', lower=1e-6, upper=1e-2, log=True, default_value=5e-4)
    lr_decay = ConfigSpace.UniformIntegerHyperparameter(
        name='lr_decay', lower=2, upper=1000, log=True, default_value=10)
    patience = ConfigSpace.UniformIntegerHyperparameter(
        name='patience', lower=2, upper=200, log=False, default_value=10)
    tolerance = ConfigSpace.UniformFloatHyperparameter(
        name='tolerance', lower=1e-5, upper=1e-2, log=True, default_value=1e-4)
    resize_crop = ConfigSpace.CategoricalHyperparameter(
        name='resize_crop', choices=[True, False], default_value=False)
    h_flip = ConfigSpace.CategoricalHyperparameter(
        name='h_flip', choices=[True, False], default_value=False)
    v_flip = ConfigSpace.CategoricalHyperparameter(
        name='v_flip', choices=[True, False], default_value=False)
    shuffle = ConfigSpace.CategoricalHyperparameter(
        name='shuffle', choices=[True, False], default_value=True)

    cs.add_hyperparameters([
        batch_size,
        learning_rate_init,
        epochs,
        momentum,
        weight_decay,
        lr_decay,
        patience,
        tolerance,
        resize_crop,
        h_flip,
        v_flip,
        shuffle,
    ])

    return cs

def get_hyperparameter_search_space_fruits(seed=None):
    """
    Neural Network search space based on a best effort using the scikit-learn
    implementation. Note that for state of the art performance, other packages
    could be preferred.

    Parameters
    ----------
    seed: int
        Random seed that will be used to sample random configurations

    Returns
    -------
    cs: ConfigSpace.ConfigurationSpace
        The configuration space object
    """
    cs = ConfigSpace.ConfigurationSpace('ResNet18_classifier', seed)
    learning_rate_init = ConfigSpace.UniformFloatHyperparameter(
        name='learning_rate_init', lower=1e-6, upper=1e-1, log=True, default_value=1e-1)
    epochs = ConfigSpace.UniformIntegerHyperparameter(
        name='epochs', lower=1, upper=50, default_value=30)
    batch_size = ConfigSpace.CategoricalHyperparameter(
        name='batch_size', choices=[8, 16, 32, 64, 128], default_value=128)
    momentum = ConfigSpace.UniformFloatHyperparameter(
        name='momentum', lower=0, upper=1, default_value=0.9)
    weight_decay = ConfigSpace.UniformFloatHyperparameter(
        name='weight_decay', lower=1e-6, upper=1e-2, log=True, default_value=5e-4)
    lr_decay = ConfigSpace.UniformIntegerHyperparameter(
        name='lr_decay', lower=2, upper=1000, log=True, default_value=10)
    patience = ConfigSpace.UniformIntegerHyperparameter(
        name='patience', lower=2, upper=200, log=False, default_value=10)
    tolerance = ConfigSpace.UniformFloatHyperparameter(
        name='tolerance', lower=1e-5, upper=1e-2, log=True, default_value=1e-4)
    resize_crop = ConfigSpace.CategoricalHyperparameter(
        name='resize_crop', choices=[True, False], default_value=False)
    h_flip = ConfigSpace.CategoricalHyperparameter(
        name='h_flip', choices=[True, False], default_value=False)
    v_flip = ConfigSpace.CategoricalHyperparameter(
        name='v_flip', choices=[True, False], default_value=False)
    shuffle = ConfigSpace.CategoricalHyperparameter(
        name='shuffle', choices=[True, False], default_value=True)

    cs.add_hyperparameters([
        batch_size,
        learning_rate_init,
        epochs,
        momentum,
        weight_decay,
        lr_decay,
        patience,
        tolerance,
        resize_crop,
        h_flip,
        v_flip,
        shuffle,
    ])

    return cs

def get_hyperparameter_search_space_stl10(seed=None):
    """
    Neural Network search space based on a best effort using the scikit-learn
    implementation. Note that for state of the art performance, other packages
    could be preferred.

    Parameters
    ----------
    seed: int
        Random seed that will be used to sample random configurations

    Returns
    -------
    cs: ConfigSpace.ConfigurationSpace
        The configuration space object
    """
    cs = ConfigSpace.ConfigurationSpace('ResNet18_classifier', seed)
    learning_rate_init = ConfigSpace.UniformFloatHyperparameter(
        name='learning_rate_init', lower=1e-6, upper=1, log=True, default_value=1e-1)
    epochs = ConfigSpace.UniformIntegerHyperparameter(
        name='epochs', lower=1, upper=200, default_value=150)
    batch_size = ConfigSpace.CategoricalHyperparameter(
        name='batch_size', choices=[8, 16, 32, 64, 128], default_value=128)
    momentum = ConfigSpace.UniformFloatHyperparameter(
        name='momentum', lower=0, upper=1, default_value=0.9)
    weight_decay = ConfigSpace.UniformFloatHyperparameter(
        name='weight_decay', lower=1e-6, upper=1e-2, log=True, default_value=5e-4)
    lr_decay = ConfigSpace.UniformIntegerHyperparameter(
        name='lr_decay', lower=2, upper=1000, log=True, default_value=10)
    patience = ConfigSpace.UniformIntegerHyperparameter(
        name='patience', lower=2, upper=200, log=False, default_value=10)
    tolerance = ConfigSpace.UniformFloatHyperparameter(
        name='tolerance', lower=1e-5, upper=1e-2, log=True, default_value=1e-4)
    resize_crop = ConfigSpace.CategoricalHyperparameter(
        name='resize_crop', choices=[True, False], default_value=False)
    h_flip = ConfigSpace.CategoricalHyperparameter(
        name='h_flip', choices=[True, False], default_value=False)
    v_flip = ConfigSpace.CategoricalHyperparameter(
        name='v_flip', choices=[True, False], default_value=False)
    shuffle = ConfigSpace.CategoricalHyperparameter(
        name='shuffle', choices=[True, False], default_value=True)

    cs.add_hyperparameters([
        batch_size,
        learning_rate_init,
        epochs,
        momentum,
        weight_decay,
        lr_decay,
        patience,
        tolerance,
        resize_crop,
        h_flip,
        v_flip,
        shuffle,
    ])

    return cs

def get_hyperparameter_search_space_flower(seed=None):
    """
    Neural Network search space based on a best effort using the scikit-learn
    implementation. Note that for state of the art performance, other packages
    could be preferred.

    Parameters
    ----------
    seed: int
        Random seed that will be used to sample random configurations

    Returns
    -------
    cs: ConfigSpace.ConfigurationSpace
        The configuration space object
    """
    cs = ConfigSpace.ConfigurationSpace('ResNet18_classifier', seed)
    learning_rate_init = ConfigSpace.UniformFloatHyperparameter(
        name='learning_rate_init', lower=1e-6, upper=1e-1, log=True, default_value=1e-1)
    epochs = ConfigSpace.UniformIntegerHyperparameter(
        name='epochs', lower=1, upper=200, default_value=150)
    batch_size = ConfigSpace.CategoricalHyperparameter(
        name='batch_size', choices=[8, 16, 32, 64, 128], default_value=128)
    momentum = ConfigSpace.UniformFloatHyperparameter(
        name='momentum', lower=0, upper=1, default_value=0.9)
    weight_decay = ConfigSpace.UniformFloatHyperparameter(
        name='weight_decay', lower=1e-6, upper=1e-2, log=True, default_value=5e-4)
    lr_decay = ConfigSpace.UniformIntegerHyperparameter(
        name='lr_decay', lower=2, upper=1000, log=True, default_value=10)
    patience = ConfigSpace.UniformIntegerHyperparameter(
        name='patience', lower=2, upper=200, log=False, default_value=10)
    tolerance = ConfigSpace.UniformFloatHyperparameter(
        name='tolerance', lower=1e-5, upper=1e-2, log=True, default_value=1e-4)
    resize_crop = ConfigSpace.CategoricalHyperparameter(
        name='resize_crop', choices=[True, False], default_value=False)
    h_flip = ConfigSpace.CategoricalHyperparameter(
        name='h_flip', choices=[True, False], default_value=False)
    v_flip = ConfigSpace.CategoricalHyperparameter(
        name='v_flip', choices=[True, False], default_value=False)
    shuffle = ConfigSpace.CategoricalHyperparameter(
        name='shuffle', choices=[True, False], default_value=True)

    cs.add_hyperparameters([
        batch_size,
        learning_rate_init,
        epochs,
        momentum,
        weight_decay,
        lr_decay,
        patience,
        tolerance,
        resize_crop,
        h_flip,
        v_flip,
        shuffle,
    ])

    return cs

dataset = ['cifar10', 'cifar100', 'fmnist', 'mnist', 'scmnist', 'dvc', 'svhn', 'fruits', 'stl10', 'flower']
response_type = ['acc', 'time']
dic = {'0':'batch_size','1':'epochs','2':'h_flip','3':'learning_rate_init','4':'lr_decay','5':'momentum','6':'patience','7':'resize_crop','8':'shuffle','9':'tolerance','10':'v_flip','11':'weight_decay'}
os.mkdir('imps')
for d in dataset:
    print(d)
    for r in response_type:

        # artificial dataset (here: features)
        features = np.loadtxt(path + '/data/12param/'+d+'/'+d+'-features.csv', delimiter=",")
        responses = np.loadtxt(path + '/data/12param/'+d+'/'+d+'-responses-'+r+'.csv', delimiter=",")

        cs = getattr(sys.modules[__name__], "get_hyperparameter_search_space_%s" % d)()

        #percentiles = np.percentile(responses, 75)
        #cutoffs = (percentiles, np.inf)
        f = fANOVA(X = features, Y = responses, config_space=cs, n_trees=16, seed=7)#, cutoffs=cutoffs)

        # marginal of particular parameter:
        fil = open('imps/imp-'+d+'.txt', 'a')
        for i in range(12):
            dims = (i, )
            res = f.quantify_importance(dims)
            fil.write(str(res)+'\n')

with open('importance-single-12params.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')#,quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['dataset', 'parameter', 'marginal_contribution_accuracy', 'marginal_contribution_runtime'])
    d = os.listdir('imps/')
    for i in d:
        #print(i)
        f = open('imps/'+i, 'r')
        lines = f.readlines()
        for line in range(12):
            words = lines[line].split()
            #print(words[0][2:4])
            words_t = lines[line+12].split()
            res = [i.split('-')[1][:-4]]
            if words[0][3] == ',':
                res.append(dic[words[0][2]])
            else:
                res.append(dic[words[0][2:4]])
            res.append(words[3][:-1])
            res.append(words_t[3][:-1])
            writer.writerow(res)
