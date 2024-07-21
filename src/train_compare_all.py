# -*- coding: utf-8 -*-
# Author : Junhao Wei
# @file : train_compare_all.py
# @Time : 2024/7/14 17:30
# Interpretation: Train to find best param-method pair for different models.
import os

from src.pkts import preprocessing_modules
from src.train_flow import pre_processing, train_ml_models
from src.pkts.preprocessing_modules import ALL_PRE_FEATURES


def main_flow_compare(data_path, use_feats_list, n_tairls, use_metric, class_weights, rif=False):
    # for compare
    using_models = ['LGBM', 'RF', 'CB']
    using_onehot = [True, False]

    # train & evaluate one model
    for one_hot in using_onehot:
        feats, labels, x_train, x_test, y_train, y_test = (
            pre_processing.get_processed_data(data_path, use_features=use_feats_list, one_hot=one_hot))
        # ROS + cross-validation for pre-train  (If rif, no cv needed)
        feats, labels = preprocessing_modules.data_resampling(feats, labels, sampling_method='ROS')

        for model in using_models:
            cur_model = model
            cur_onehot = one_hot

            train_ml_models.train_single_model_sampling(feats, labels,
                                                        x_train, x_test, y_train, y_test,
                                                        cur_model, n_tairls, cur_onehot,
                                                        use_metric, class_weights)


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             'data', 'Combined', 'CombineNassCiss.csv')
    print(data_path)
    use_feats_list = ALL_PRE_FEATURES

    # for pre-train
    n_trials = 200
    using_metric = 'f1-macro'
    given_class_weights = {0: 1, 1: 1, 2: 5}  # Only used if resampling.

    main_flow_compare(data_path, use_feats_list, n_trials, using_metric, given_class_weights)

# TODO: Modify this file (class-weights part mainly).
# TODO: flow: run pre-processing -> (generate new train, test data) rif new data -> pre-train with rif_rus -> retrain
# retrain set: (None, over-sampling techs, rif+rus) × (no_oh, oh)

# TODO: delete train_ml_modules.py and make pre-train, retrain as 2 folds in this file!
