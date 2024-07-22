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
        feats, labels = preprocessing_modules.data_resampling(feats, labels, sampling_method='ROS')

        for model in using_models:
            # Pretrain for hyper-params tuning
            best_params, _, result_dir = param_search_by_optuna(feats, labels, study_name=model,
                                                                metric=use_metric, n_trials=n_trials, one_hot=one_hot)

    # Pre-train finished, retrain with several settings
    # rif_setting = [True, False]
    # sampling_setting = ['ROS', 'ADASYN', 'SMOTETomek', 'None']
    # pre_trained_params_path = os.listdir(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(data_path))),
    #                                                   'trained_model_info'))
    # for model_name in pre_trained_params_path:
    #     for rif_set in rif_setting:
    #         if rif_set:
    #             x_train_rif_rus = pd.read_csv()
    #             y_train_rif_rus = 

    # Retrain for performance evaluation
    # rif_str = 'with_rif' if rif else 'no_rif'
    # un_samp = 'None'
    # if rif:
    #     un_samp = 'TomekLinks'  # add: Never use None, use rus and tl! (data prepared)
    #     x_train, y_train, x_test = retrain_modules.prepare_rif_setting(x_train, y_train, x_test)
    # use_sampling = un_samp if un_samp != 'None' else sampling
    # retrain_modules.retrain(model, best_params, x_train, y_train, x_test, y_test, sampling=use_sampling,
    #                         class_weights=class_weights, save_dir=result_dir, rifed=rif_str)


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                             'data', 'Combined', 'CombineNassCiss.csv')
    use_feats_list = ALL_PRE_FEATURES

    # for pre-train
    trials = 400
    using_metric = 'f1-macro'
    # given_class_weights = {0: 1, 1: 1, 2: 5}  # Only used if resampling.

    main_flow_pre_train(data_path, use_feats_list, trials, using_metric)

# TODO: Modify this file (class-weights part mainly).
# TODO: flow: run pre-processing -> (generate new train, test data) rif new data -> pre-train with rif_rus -> retrain
# retrain set: (None, over-sampling techs, rif+rus) × (no_oh, oh)

# TODO: delete train_ml_modules.py and make pre-train, retrain as 2 folds in this file!
