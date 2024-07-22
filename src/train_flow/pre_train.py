# -*- coding: utf-8 -*-
# Author : Junhao Wei
# @file : pre_train.py
# @Time : 2024/7/14 17:30
# Interpretation: Train to find best param-method pair for different models.
import os

import pandas as pd

from src.pkts import preprocessing_modules
from src.train_flow import pre_processing
from src.pkts.preprocessing_modules import ALL_PRE_FEATURES
from src.pkts.pretrain_modules import param_search_by_optuna


def main_flow_pre_train(data_path, use_feats_list, n_trials, use_metric):
    # for compare
    using_models = ['LGBM', 'RF', 'CB']
    using_onehot = [True, False]

    # pre-train models by ROS with/without one-hot setting (6-fold CV used)
    for one_hot in using_onehot:
        feats, labels, x_train, x_test, y_train, y_test = (
            pre_processing.get_processed_data(data_path, use_features=use_feats_list, one_hot=one_hot))
        # ROS + cross-validation for pre-train  (If rif, no cv needed)
        for model in using_models:
            # Pretrain for hyper-params tuning
            best_params, _, result_dir = param_search_by_optuna(feats, labels, study_name=model,
                                                                metric=use_metric, n_trials=n_trials, one_hot=one_hot)


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                             'data', 'Combined', 'CombineNassCiss.csv')
    use_feats_list = ALL_PRE_FEATURES

    # for pre-train
    trials = 200
    using_metric = 'f1-macro'
    # given_class_weights = {0: 1, 1: 1, 2: 5}  # Only used if resampling.

    main_flow_pre_train(data_path, use_feats_list, trials, using_metric)
