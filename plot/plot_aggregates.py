import argparse
import logging
import math
import matplotlib
import matplotlib.pyplot as plt
import Orange
import os
import pandas as pd
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='../data/aggregated/importance-single.csv')
    parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~/experiments/cnn_fanova/'))
    return parser.parse_args()


def critical_dist(numModels, numDatasets):
    # confidence values for alpha = 0.05. Index is the number of models (minimal two)
    alpha005 = [-1, -1, 1.959964233, 2.343700476, 2.569032073, 2.727774717,
                2.849705382, 2.948319908, 3.030878867, 3.10173026, 3.16368342,
                3.218653901, 3.268003591, 3.312738701, 3.353617959, 3.391230382,
                3.426041249, 3.458424619, 3.488684546, 3.517072762, 3.543799277,
                3.569040161, 3.592946027, 3.615646276, 3.637252631, 3.657860551,
                3.677556303, 3.696413427, 3.71449839, 3.731869175, 3.748578108,
                3.764671858, 3.780192852, 3.795178566, 3.809663649, 3.823679212,
                3.837254248, 3.850413505, 3.863181025, 3.875578729, 3.887627121,
                3.899344587, 3.910747391, 3.921852503, 3.932673359, 3.943224099,
                3.953518159, 3.963566147, 3.973379375, 3.98296845, 3.992343271,
                4.001512325, 4.010484803, 4.019267776, 4.02786973, 4.036297029,
                4.044556036, 4.05265453, 4.060596753, 4.068389777, 4.076037844,
                4.083547318, 4.090921028, 4.098166044, 4.105284488, 4.112282016,
                4.119161458, 4.125927056, 4.132582345, 4.139131568, 4.145576139,
                4.151921008, 4.158168297, 4.164320833, 4.170380738, 4.176352255,
                4.182236797, 4.188036487, 4.19375486, 4.199392622, 4.204952603,
                4.21043763, 4.215848411, 4.221187067, 4.22645572, 4.23165649,
                4.236790793, 4.241859334, 4.246864943, 4.251809034, 4.256692313,
                4.261516196, 4.266282802, 4.270992841, 4.275648432, 4.280249575,
                4.284798393, 4.289294885, 4.29374188, 4.298139377, 4.302488791]
    return alpha005[numModels] * math.sqrt((numModels * (numModels + 1)) / (6 * numDatasets))


def df_sorted(df, by, column):
    medians = pd.DataFrame({col: vals[column] for col, vals in df.groupby(by)}).median()
    df['median'] = df.apply(lambda row: medians[row['parameter']], axis=1)
    df = df.sort_values('median')
    del df['median']
    return df


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)

    matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]
    params = {'text.usetex': True,
              'font.size': 11,
              'font.family': 'lmodern',
              'text.latex.unicode': True,
              }
    matplotlib.rcParams.update(params)
    iWidth = 6
    iTextspace = 1.3

    data = pd.read_csv(args.input_file)
    for column in ['marginal_contribution_accuracy', 'marginal_contribution_runtime']:
        measure = column.split('_')[-1]

        # plot marginal contribution boxplot
        data_measure = df_sorted(data, ['parameter'], 'marginal_contribution_accuracy')
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(x='parameter', y=column, data=data_measure, ax=ax)
        output_file = os.path.join(args.output_dir, 'boxplot_%s.png' % measure)
        ax.set_ylabel(measure)
        plt.savefig(output_file)
        logging.info('saved to %s' % output_file)

        # plot Nemenyi
        data_pivot = data.pivot(index='parameter', columns='dataset', values=column)
        num_params, num_datasets = data_pivot.shape
        data_rank = data_pivot.rank(axis=0, method='average', ascending=False)
        average_ranks = data_rank.sum(axis=1) / num_datasets
        average_ranks_dict = average_ranks.to_dict()
        cd = critical_dist(num_params, num_datasets)
        output_file = os.path.join(args.output_dir, 'nemenyi_%s.png' % measure)
        Orange.evaluation.scoring.graph_ranks(list(average_ranks_dict.values()),
                                              list(average_ranks_dict.keys()),
                                              cd=cd, filename=output_file,
                                              width=iWidth, textspace=iTextspace)
        logging.info('saved to %s' % output_file)


if __name__ == '__main__':
    run(parse_args())
