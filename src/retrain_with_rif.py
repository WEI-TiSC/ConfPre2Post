# -*- coding: utf-8 -*-
# Author : Junhao Wei
# @file : retrain_with_rif.py
# @Time : 2024/7/19 14:10
# Interpretation
import os
import json

import pandas as pd

from src.pkts import retrain_modules

if __name__ == "__main__":
    models_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               'trained_model_info')

    one_hot_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                     'data', 'Combined', 'with_one_hot')
    no_onehot_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                     'data', 'Combined', 'no_one_hot')

    # ----------------------------------------------------------------------
    # Check if rif-ed file exists
    x_noh_train_rif_path = os.path.join(no_onehot_path, 'x_train_rif.csv')
    y_noh_train_fif_path = os.path.join(no_onehot_path, 'y_train_rif.csv')

    if not os.path.exists(x_noh_train_rif_path):
        x_noh_train = pd.read_csv(os.path.join(no_onehot_path, 'x_train.csv'))
        y_noh_train = pd.read_csv(os.path.join(no_onehot_path, 'y_train.csv'))

        x_noh_train, y_noh_train = retrain_modules.prepare_rif_setting(x_noh_train, y_noh_train)
        x_noh_train.to_csv(os.path.join(no_onehot_path, 'x_train_rif.csv'), index=False)
        y_noh_train.to_csv(os.path.join(no_onehot_path, 'y_train_rif.csv'), index=False)
    else:
        x_noh_train = pd.read_csv(x_noh_train_rif_path)
        y_noh_train = pd.read_csv(y_noh_train_fif_path)
        y_noh_train = pd.Series(y_noh_train['InjurySeverity'].values)

    x_noh_test = pd.read_csv(os.path.join(no_onehot_path, 'x_test.csv'))
    y_noh_test = pd.read_csv(os.path.join(no_onehot_path, 'y_test.csv'))
    y_noh_test = pd.Series(y_noh_test['InjurySeverity'].values)

    if 'CASEWGT' in x_noh_test.columns.values:
        x_noh_test = x_noh_test.drop(columns=['CASEWGT'])

    # ----------------------------------------------------------------------
    # Check if rif-ed file exists
    x_oh_train_rif_path = os.path.join(one_hot_data_path, 'x_train_rif.csv')
    y_oh_train_fif_path = os.path.join(one_hot_data_path, 'y_train_rif.csv')

    if not os.path.exists(x_oh_train_rif_path):
        x_oh_train = pd.read_csv(os.path.join(one_hot_data_path, 'x_train.csv'))
        y_oh_train = pd.read_csv(os.path.join(one_hot_data_path, 'y_train.csv'))

        x_oh_train, y_oh_train = retrain_modules.prepare_rif_setting(x_oh_train, y_oh_train)
        x_oh_train.to_csv(os.path.join(one_hot_data_path, 'x_train_rif.csv'), index=False)
        y_oh_train.to_csv(os.path.join(one_hot_data_path, 'y_train_rif.csv'), index=False)
    else:
        x_oh_train = pd.read_csv(x_oh_train_rif_path)
        y_oh_train = pd.read_csv(y_oh_train_fif_path)
        y_oh_train = pd.Series(y_oh_train['InjurySeverity'].values)

    x_oh_test = pd.read_csv(os.path.join(one_hot_data_path, 'x_test.csv'))
    y_oh_test = pd.read_csv(os.path.join(one_hot_data_path, 'y_test.csv'))
    y_oh_test = pd.Series(y_oh_test['InjurySeverity'].values)

    if 'CASEWGT' in x_oh_test.columns.values:
        x_noh_test = x_oh_test.drop(columns=['CASEWGT'])
    # ----------------------------------------------------------------------
    # Set up retrain flow
    models_name = [each for each in os.listdir(models_path)]
    sampling_range = ['None', 'TomekLinks']

    for model in models_name:
        model_param_dir_list = [x for x in os.listdir(os.path.join(models_path, model))]
        cur_save_dir = os.path.join(models_path, model)
        for ml_setting in model_param_dir_list:
            pre_train_params = os.path.join(models_path, model, ml_setting, 'pre_train_info', 'param_dict.json')
            with open(pre_train_params, 'r', encoding='utf-8') as f:
                param_dict = json.load(fp=f)

            # Get sure about whether one-hot
            if 'NoOnehot' in ml_setting:
                for sampling in sampling_range:
                    retrain_modules.retrain(model, param_dict, x_noh_train, y_noh_train, x_noh_test, y_noh_test,
                                            sampling=sampling, rifed='with_rif', class_weights={0: 1, 1: 1, 2: 6},
                                            save_dir=os.path.join(cur_save_dir, ml_setting))
            else:
                for sampling in sampling_range:
                    retrain_modules.retrain(model, param_dict, x_oh_train, y_oh_train, x_oh_test, y_oh_test,
                                            sampling=sampling, rifed='with_rif', class_weights={0: 1, 1: 1, 2: 6},
                                            save_dir=os.path.join(cur_save_dir, ml_setting))
