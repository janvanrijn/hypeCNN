import argparse
from hypecnn import config_spaces
import logging
import numpy as np
import fanova
import os
path = os.path.dirname(os.path.realpath(__file__))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='data/12param/')
    parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~/experiments/cnn_fanova/importances'))
    return parser.parse_args()


def run(args):
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    os.makedirs(args.output_dir, exist_ok=True)

    legal_hyperparameters = None
    for dataset in config_spaces.DATASETS:
        logging.info('Dataset %s' % dataset)

        # artificial dataset (here: features)
        directory = os.path.join(args.input_dir, dataset)
        features = np.loadtxt(directory + '/' + dataset + '-features.csv', delimiter=",")
        responses = np.loadtxt(directory + '/' + dataset + '-responses-acc.csv', delimiter=",")

        cs = config_spaces.get_config_space(dataset, 0)
        if legal_hyperparameters is None:
            legal_hyperparameters = cs.get_hyperparameter_names()
        else:
            if legal_hyperparameters != cs.get_hyperparameter_names():
                raise ValueError()

        fanova_model = fanova.fANOVA(
            X=features, Y=responses, config_space=cs, n_trees=16, seed=7
        )

        # marginal of particular parameter:
        output_file = os.path.join(args.output_dir, '%s.txt' % dataset)
        with open(output_file, 'w') as fp:
            for idx, hyperparameter in enumerate(legal_hyperparameters):
                logging.info('Hyperparameter %d: %s' % (idx, hyperparameter))
                dims = (idx, )
                res = fanova_model.quantify_importance(dims)
                fp.write(str(res)+'\n')


if __name__ == '__main__':
    run(parse_args())
