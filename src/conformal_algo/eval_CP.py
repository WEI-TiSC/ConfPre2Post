import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def calc_coverage(conf_set, y_true):
    """
    Calculate coverage.

    :param conf_set: conformal prediction set.
    :param y_true: real labels.

    :return: coverage (float in [0, 1]).
    """
    involved_sum = 0
    for i in range(len(conf_set)):
        if y_true in conf_set[i].keys():
            involved_sum += 1

    return round(involved_sum / len(y_true), 3)


def calc_average_set_size(conf_set):
    """
    Calculate average prediction set size.

    :param conf_set: conformal prediction set.

    :return: average set size.
    """
    total_prediction_num = 0
    for i in range(len(conf_set)):
        total_prediction_num += len(conf_set[i])
    avg_size = round(total_prediction_num / len(conf_set), 3)
    return avg_size


def calc_avg_set_size_by_class(conf_set, y_true, n_class=3):
    total_size = [0 for _ in range(n_class)]
    total_cases = [0 for _ in range(n_class)]
    avg_size_by_class = [0 for _ in range(n_class)]

    for i, val in enumerate(y_true):
        total_cases[val] += 1
        total_size[val] += len(conf_set[i])

    avg_size_all = round(sum(total_size)/sum(total_cases), 2)
    for i in range(n_class):
        avg_size_by_class[i] = round(total_size[i]/total_cases[i], 2)
    return avg_size_all, avg_size_by_class


def calc_ssc_metric(conf_set, y_true):
    """
    ssc = total_involved_samples_num / total_prediction_set_size

    """
    involved_sum = 0
    total_set_size = 0

    for i, val in enumerate(y_true):
        total_set_size += len(conf_set[i])
        if val in conf_set[i].keys():
            involved_sum += 1

    ssc_metric = round(involved_sum / total_set_size, 3)
    return ssc_metric


def draw_set_sizes(conf_set, cp_type: str, save_path, model_info, alpha, serious_only=False):
    """
    Draw set size figure.

    :param conf_set: conformal prediction set.
    :param cp_type: type of conformal threshold.
    """

    sets_dim = [len(pred_set) for pred_set in conf_set]
    set_size_dict = dict(pd.Series(sets_dim).value_counts())
    plt.bar(set_size_dict.keys(), set_size_dict.values())
    plt.xlabel('Set Size')
    plt.ylabel('Counts')
    x_tick = [1, 2, 3]
    plt.xticks(x_tick)
    set_sizes = sorted(set_size_dict.items(), key = lambda item: item[0])
    for idx, set_size in zip(x_tick, set_sizes):
        plt.text(idx, set_size[1], set_size[1], ha='center', va='bottom', fontsize=10)

    serious = ''
    if serious_only:
        serious = 'Serious Injury of '

    plt.title(f"Set Size Distribution of {serious} {cp_type} method")

    plt.savefig(
        os.path.join(save_path, f'Set size distribution of {serious} {model_info} under {1 - alpha} confidence level.png'), dpi=300)
    plt.clf()
    # plt.show()


def draw_coverage(conf_set, y_true, cp_type: str, save_path, model_info, alpha):
    """
    Draw coverage figure.

    :param conf_set: conformal prediction set.
    :param y_true: true labels of corresponding set.
    :param cp_type: type of conformal threshold.
    :param total_classes: total classes number in problem.
    """
    total_classes = len(y_true.value_counts())
    class_cases = np.zeros(total_classes)
    covered_cases = np.zeros(total_classes)
    cover_rate = np.zeros(total_classes)

    y_true = y_true.tolist()
    for i in range(len(conf_set)):
        class_cases[y_true[i]] += 1
        if y_true[i] in conf_set[i].keys():
            covered_cases[y_true[i]] += 1

    x_axis = ['No Injury', 'Slight Injury', 'Serious Injury']

    for i in range(total_classes):
        cover_rate[i] = covered_cases[i] / class_cases[i]
    idx_len = np.arange(total_classes)

    plt.bar(x_axis, covered_cases / class_cases)
    for idx, covered in zip(idx_len, cover_rate):
        plt.text(idx, covered, '%.3f' % covered, ha='center', va='bottom', fontsize=10)

    plt.axhline(y=0.9, color='r', linestyle='--')
    plt.title(f'{cp_type} method coverage')
    plt.xlabel('Classes')
    plt.ylabel('Coverage')

    plt.savefig(os.path.join(save_path, f'Class-wise coverage of {model_info} under {1 - alpha} confidence level.png'),
                dpi=300)
    plt.clf()
    # plt.show()
