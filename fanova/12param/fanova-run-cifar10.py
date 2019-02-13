import matplotlib
matplotlib.use('Agg')
import numpy as np
from fanova import fANOVA
import fanova.visualizer
import matplotlib.pyplot as plt
import ConfigSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

import os
path = os.path.dirname(os.path.realpath(__file__))

response_type = 'time'

# directory in which you can find all plots
plot_dir = path + '/test_plots_'+response_type

# artificial dataset (here: features)
features = np.loadtxt(path + '/cifar10-features.csv', delimiter=",")
responses = np.loadtxt(path + '/cifar10-responses-'+response_type+'.csv', delimiter=",")

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
for i in range(12):
	vis.plot_marginal(i, show=False)
	plt.savefig(plot_dir+'/'+str(i)+'.png')
	plt.clf()
