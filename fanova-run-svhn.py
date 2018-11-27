import numpy as np
from fanova import fANOVA
import fanova.visualizer
import matplotlib.pyplot as plt
import ConfigSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

import os
path = os.path.dirname(os.path.realpath(__file__))

response_type = 'acc' #'time'

# directory in which you can find all plots
plot_dir = path + '/data/svhn/test_plots_'+response_type

# artificial dataset (here: features)
features = np.loadtxt(path + '/data/svhn/svhn-features.csv', delimiter=",")
responses = np.loadtxt(path + '/data/svhn/svhn-responses-'+response_type+'.csv', delimiter=",")

def get_hyperparameter_search_space(seed=None):
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
    # batch_size = ConfigSpace.UniformIntegerHyperparameter(
    #     name='batch_size', lower=1, upper=256, log=True, default_value=128)
    # learning_rate = ConfigSpace.CategoricalHyperparameter(
    #     name='learning_rate', choices=['constant', 'invscaling', 'adaptive'], default_value='constant')
    learning_rate_init = ConfigSpace.UniformFloatHyperparameter(
        name='learning_rate_init', lower=1e-6, upper=1, log=True, default_value=1e-1)

    epochs = ConfigSpace.UniformIntegerHyperparameter(
        name='epochs', lower=1, upper=400, default_value=300)
    batch_size = ConfigSpace.CategoricalHyperparameter(
        name='batch_size', choices=[32, 64, 128, 256, 512], default_value=128)
    # shuffle = ConfigSpace.CategoricalHyperparameter(
    #     name='shuffle', choices=[True, False], default_value=True)
    momentum = ConfigSpace.UniformFloatHyperparameter(
        name='momentum', lower=0, upper=1, default_value=0.9)
    weight_decay = ConfigSpace.UniformFloatHyperparameter(
        name='weight_decay', lower=1e-6, upper=1e-2, log=True, default_value=5e-4)

    cs.add_hyperparameters([
        batch_size,
        learning_rate_init,
        epochs,
        # shuffle,
        momentum,
        weight_decay,
    ])

    return cs

cs = get_hyperparameter_search_space()

# create an instance of fanova with trained forest and ConfigSpace
f = fANOVA(X = features, Y = responses, config_space=cs, n_trees=16, seed=7)

# marginal of particular parameter:
# dims = (1, )
# res = f.quantify_importance(dims)
# print(res)

# visualizations:
# first create an instance of the visualizer with fanova object and configspace
vis = fanova.visualizer.Visualizer(f, cs, plot_dir)
# plot marginals for each parameter
for i in range(5):
	vis.plot_marginal(i, show=False)
	plt.savefig(plot_dir+'/'+str(i)+'.png')
	plt.clf()
