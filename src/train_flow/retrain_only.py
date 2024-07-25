# -*- coding: utf-8 -*-
# Author : Junhao Wei
# @file : retrain_only.py
# @Time : 2024/7/19 14:10
# Interpretation: 3. retrain from pre-trained params
import json
import os

import pandas as pd

from src.pkts import retrain_modules, preprocessing_modules

if __name__ == "__main__":
    models_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                               'trained_model_info')

    one_hot_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                     'data', 'Combined', 'with_one_hot')
    no_onehot_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                  'data', 'Combined', 'no_one_hot')

    # ----------------------------------------------------------------------
    # # Check if rif-ed file exists
    # x_noh_train_rif_path = os.path.join(no_onehot_path, 'x_train_rif_rus.csv')
    # y_noh_train_fif_path = os.path.join(no_onehot_path, 'y_train_rif_rus.csv')
    #
    # # No rif data
    # x_noh_train = pd.read_csv(os.path.join(no_onehot_path, 'x_train.csv'))
    # y_noh_train = pd.read_csv(os.path.join(no_onehot_path, 'y_train.csv'))
    #
    # if not os.path.exists(x_noh_train_rif_path):
    #     # rif setting
    #     # x_noh_train_rif, y_noh_train_rif = retrain_modules.prepare_rif_setting(x_noh_train, y_noh_train)
    #     # x_noh_train_rif.to_csv(os.path.join(no_onehot_path, 'x_train_rif.csv'), index=False)
    #     # y_noh_train_rif.to_csv(os.path.join(no_onehot_path, 'y_train_rif.csv'), index=False)
    #     x_noh_train_rif = pd.read_csv(os.path.join(no_onehot_path, 'x_train_rif.csv'))
    #     y_noh_train_rif = pd.read_csv(os.path.join(no_onehot_path, 'y_train_rif.csv'))
    #
    #     # RUS sampling
    #     x_noh_train_rif_rus, y_noh_train_rif_rus = preprocessing_modules.data_resampling(x_noh_train_rif, y_noh_train_rif, sampling_method='RUS')
    #     x_noh_train_rif_rus.to_csv(os.path.join(no_onehot_path, 'x_train_rif_rus.csv'), index=False)
    #     y_noh_train_rif_rus.to_csv(os.path.join(no_onehot_path, 'y_train_rif_rus.csv'), index=False)
    #
    # else:
    #     x_noh_train_rif_rus = pd.read_csv(x_noh_train_rif_path)
    #     y_noh_train_rif_rus = pd.read_csv(y_noh_train_fif_path)
    #     q1 = y_noh_train_rif_rus.value_counts()
    #     y_noh_train_rif_rus = pd.Series(y_noh_train_rif_rus['InjurySeverity'].values)
    #
    # x_noh_test = pd.read_csv(os.path.join(no_onehot_path, 'x_test.csv'))
    # y_noh_test = pd.read_csv(os.path.join(no_onehot_path, 'y_test.csv'))
    # y_noh_train = pd.Series(y_noh_train['InjurySeverity'].values)
    # y_noh_test = pd.Series(y_noh_test['InjurySeverity'].values)
    #
    # if 'CASEWGT' in x_noh_test.columns.values:
    #     x_noh_test = x_noh_test.drop(columns=['CASEWGT'])
    # if 'CASEWGT' in x_noh_train.columns.values:
    #     x_noh_train = x_noh_train.drop(columns=['CASEWGT'])

    # ----------------------------------------------------------------------
    # Check if rif-ed file exists
    x_oh_train_rif_path = os.path.join(one_hot_data_path, 'x_train_rif_rus.csv')
    y_oh_train_fif_path = os.path.join(one_hot_data_path, 'y_train_rif_rus.csv')

    # No rif data
    x_oh_train = pd.read_csv(os.path.join(one_hot_data_path, 'x_train.csv'))
    y_oh_train = pd.read_csv(os.path.join(one_hot_data_path, 'y_train.csv'))

    # Prepare rif setting
    if not os.path.exists(x_oh_train_rif_path):
        x_oh_train_rif, y_oh_train_rif = retrain_modules.prepare_rif_setting(x_oh_train, y_oh_train)
        x_oh_train_rif.to_csv(os.path.join(one_hot_data_path, 'x_train_rif.csv'), index=False)
        y_oh_train_rif.to_csv(os.path.join(one_hot_data_path, 'y_train_rif.csv'), index=False)
        # x_oh_train_rif = pd.read_csv(os.path.join(one_hot_data_path, 'x_train_rif.csv'))
        # y_oh_train_rif = pd.read_csv(os.path.join(one_hot_data_path, 'y_train_rif.csv'))

        # RUS sampling
        x_oh_train_rif_rus, y_oh_train_rif_rus = preprocessing_modules.data_resampling(x_oh_train_rif, y_oh_train_rif, sampling_method='RUS')
        x_oh_train_rif_rus.to_csv(os.path.join(one_hot_data_path, 'x_train_rif_rus.csv'), index=False)
        y_oh_train_rif_rus.to_csv(os.path.join(one_hot_data_path, 'y_train_rif_rus.csv'), index=False)

    else:
        x_oh_train_rif_rus = pd.read_csv(x_oh_train_rif_path)
        y_oh_train_rif_rus = pd.read_csv(y_oh_train_fif_path)
        q2 = y_oh_train_rif_rus.value_counts()
        y_oh_train_rif_rus = pd.Series(y_oh_train_rif_rus['InjurySeverity'].values)

    x_oh_test = pd.read_csv(os.path.join(one_hot_data_path, 'x_test.csv'))
    y_oh_test = pd.read_csv(os.path.join(one_hot_data_path, 'y_test.csv'))
    y_oh_train = pd.Series(y_oh_train['InjurySeverity'].values)
    y_oh_test = pd.Series(y_oh_test['InjurySeverity'].values)

    if 'CASEWGT' in x_oh_test.columns.values:
        x_oh_test = x_oh_test.drop(columns=['CASEWGT'])
    if 'CASEWGT' in x_oh_train.columns.values:
        x_oh_train = x_oh_train.drop(columns=['CASEWGT'])

    # ----------------------------------------------------------------------
    # Set up retrain flow
    class_weights = [{0: 1, 1: 1, 2: 2}, {0: 1, 1: 1, 2: 3}, {0: 1, 1: 1, 2: 4}, {0: 1, 1: 1, 2: 5}]
    models_name = [each for each in os.listdir(models_path)]
    rif_setting = [True, False]  # TODO: Only RIF+RUS available, whether to add TomekLinks(10~ day costs)?
    sampling_range = ['ROS', 'ADASYN', 'SMOTETomek']
    # sampling_range = ['None']
    # one_hot_setting = [True, False]
    one_hot_setting = [True]
    for model in models_name:
        if model == 'TabNet':
            continue
        model_param_dir_list = [x for x in os.listdir(os.path.join(models_path, model))]
        cur_save_dir = os.path.join(models_path, model)

        for ml_setting in model_param_dir_list:
            if not ml_setting.startswith('2024-07-24_Onehot'):
                continue
            pre_train_params = os.path.join(models_path, model, ml_setting, 'pre_train_info', 'param_dict.json')
            with open(pre_train_params, 'r', encoding='utf-8') as f:
                param_dict = json.load(fp=f)

            # Get sure about whether one-hot
            for cls_wgt in class_weights:
                for one_hot in one_hot_setting:
                    if one_hot:
                        for rif_set in rif_setting:
                            if rif_set:
                                retrain_modules.retrain(model, param_dict, x_oh_train_rif_rus, y_oh_train_rif_rus,
                                                        x_oh_test, y_oh_test,
                                                        sampling='None', rifed='rifed_RUS', class_weights=cls_wgt,
                                                        save_dir=os.path.join(cur_save_dir, ml_setting))
                            else:
                                for sampling in sampling_range:
                                    retrain_modules.retrain(model, param_dict, x_oh_train, y_oh_train, x_oh_test, y_oh_test,
                                                            sampling=sampling, rifed='no_rif', class_weights=cls_wgt,
                                                            save_dir=os.path.join(cur_save_dir, ml_setting))

                # else:
                #     for rif_set in rif_setting:
                #         if rif_set:
                #             retrain_modules.retrain(model, param_dict, x_noh_train_rif_rus, y_noh_train_rif_rus,
                #                                     x_noh_test, y_noh_test,
                #                                     sampling='None', rifed='rifed_RUS', class_weights=class_weights,
                #                                     save_dir=os.path.join(cur_save_dir, ml_setting))
                #         else:
                #             for sampling in sampling_range:
                #                 retrain_modules.retrain(model, param_dict, x_noh_train, y_noh_train,
                #                                         x_noh_test, y_noh_test,
                #                                         sampling=sampling, rifed='no_rif', class_weights=class_weights,
                #                                         save_dir=os.path.join(cur_save_dir, ml_setting))
