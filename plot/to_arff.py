import arff
import argparse
import ConfigSpace
import config_spaces
import logging
import numpy as np
import openmlcontrib
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='../data/12param/')
    parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~/experiments/cnn_fanova/'))
    return parser.parse_args()


def run(args):
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    column_header = [
        'batch_size', 'epochs', 'h_flip',
        'learning_rate_init', 'lr_decay', 'momentum',
        'patience', 'resize_crop', 'shuffle',
        'tolerance', 'v_flip', 'weight_decay'
    ]

    all_results = None
    for dataset in config_spaces.DATASETS:
        results = pd.read_csv(os.path.join(args.input_dir, dataset, '%s-features.csv' % dataset), header=None, names=column_header)
        accuracy = np.loadtxt(os.path.join(args.input_dir, dataset, '%s-responses-acc.csv' % dataset), delimiter=',')
        runtime = np.loadtxt(os.path.join(args.input_dir, dataset, '%s-responses-time.csv' % dataset), delimiter=',')
        assert results.shape[0] == accuracy.shape[0] == runtime.shape[0]
        results['accuracy'] = accuracy
        results['runtime'] = runtime
        results['dataset'] = dataset

        config_space = config_spaces.get_config_space(dataset, 0)
        # sanity checks on parameter values
        for hp in config_space.get_hyperparameters():
            if isinstance(hp, ConfigSpace.CategoricalHyperparameter):
                results[hp.name] = results[hp.name].apply(lambda val: hp.choices[val])
            elif isinstance(hp, ConfigSpace.hyperparameters.NumericalHyperparameter):
                for idx, value in enumerate(results[hp.name].values):
                    if not (hp.lower <= value <= hp.upper):
                        raise ValueError('Illegal value for %s at %d: %s' % (hp.name, idx, value))
            else:
                raise ValueError('Hyperparameter type not supported: %s' % hp.name)
        for idx, value in enumerate(results['accuracy'].values):
            assert 0.0 <= value < 100.0, 'Accuracy iteration %d for dataset %s: %f' % (idx, dataset, value)
        for idx, value in enumerate(results['runtime'].values):
            assert 0.0 < value
        if all_results is None:
            all_results = results
        else:
            all_results = all_results.append(results)
    os.makedirs(args.output_dir, exist_ok=True)
    arff_dict = openmlcontrib.meta.dataframe_to_arff(all_results, 'fanova-cnn', None)
    output_file = os.path.join(args.output_dir, 'fanova-cnn.arff')
    with open(output_file, 'w') as fp:
        arff.dump(arff_dict, fp)
    logging.info('saved to %s' % output_file)


if __name__ == '__main__':
    run(parse_args())
