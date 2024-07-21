# -*- coding: utf-8 -*-
# Author : Junhao Wei
# @file : train_compare_without_rif.py
# @Time : 2024/7/14 17:30
# Interpretation: Train to find best param-method pair for different models.
import os

from src.train_flow import pre_processing, train_ml_models
from src.pkts.preprocessing_modules import ALL_PRE_FEATURES


def main_flow_compare(data_path, use_feats_list, n_tairls, use_metric, class_weights, rif=False):
    # for compare
    using_models = ['LGBM', 'RF', 'CB']
    using_resampling = ['ADASYN', 'SMOTETomek', 'None']
    using_onehot = [1, 0]
    # using_models = ['LGBM']
    # using_resampling = ['None']
    # using_onehot = [1]

    feats, labels, x_train, x_test, y_train, y_test = (
        pre_processing.get_processed_data(data_path, use_features=use_feats_list))

    # train & evaluate one model
    for model in using_models:
        for sampling in using_resampling:
            for one_hot in using_onehot:
                cur_model = model
                cur_sampling = sampling
                cur_onehot = one_hot

                train_ml_models.train_single_model_sampling(feats, labels,
                                                            x_train, x_test, y_train, y_test,
                                                            cur_model, cur_sampling, n_tairls, cur_onehot,
                                                            use_metric, class_weights, rif)


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             'data', 'Combined', 'CombineNassCiss.csv')
    print(data_path)
    use_feats_list = ALL_PRE_FEATURES

    # for pre-train
    n_trials = 300
    using_metric = 'f1-macro'
    given_class_weights = {0: 1, 1: 1, 2: 5}  # Only used if resampling.

    main_flow_compare(data_path, use_feats_list, n_trials, using_metric, given_class_weights)
