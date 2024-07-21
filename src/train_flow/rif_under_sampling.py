# -*- coding: utf-8 -*-
# Author : Junhao Wei
# @file : rif_under_sampling.py
# @Time : 2024/7/21 22:53
# Interpretation

import os
import time

import pandas as pd

from src.pkts import pretrain_modules as prepro_modules, retrain_modules

# ------------------------------------------------------------------------------------------
# no_onehot_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
#                               'data', 'Combined', 'no_one_hot')
# x_noh_train_rif_path = os.path.join(no_onehot_path, 'x_train_rif.csv')
# y_noh_train_fif_path = os.path.join(no_onehot_path, 'y_train_rif.csv')
# x_noh_train = pd.read_csv(x_noh_train_rif_path)
# y_noh_train = pd.read_csv(y_noh_train_fif_path)
# y_noh_train = pd.Series(y_noh_train['InjurySeverity'].values)
# if os.path.exists(x_noh_train_rif_path):
#     x_oh_train = pd.read_csv(x_noh_train_rif_path)
#     y_oh_train = pd.read_csv(y_noh_train_fif_path)
# else:
#     x_noh_train_no_rif = pd.read_csv(os.path.join(no_onehot_path, 'x_train.csv'))
#     y_noh_train_no_rif = pd.read_csv(os.path.join(no_onehot_path, 'y_train.csv'))
#     x_noh_train, y_noh_train = retrain_modules.prepare_rif_setting(x_noh_train_no_rif, y_noh_train_no_rif)
#     x_noh_train.to_csv(os.path.join(no_onehot_path, 'x_train_rif.csv'), index=False)
#     y_noh_train.to_csv(os.path.join(no_onehot_path, 'y_train_rif.csv'), index=False)
#
# y_oh_train = pd.Series(y_oh_train['InjurySeverity'].values)

# ------------------------------------------------------------------------------------------
one_hot_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                 'data', 'Combined', 'with_one_hot')
x_oh_train_rif_path = os.path.join(one_hot_data_path, 'x_train_rif.csv')
y_oh_train_fif_path = os.path.join(one_hot_data_path, 'y_train_rif.csv')
if os.path.exists(x_oh_train_rif_path):
    x_oh_train = pd.read_csv(x_oh_train_rif_path)
    y_oh_train = pd.read_csv(y_oh_train_fif_path)
else:
    x_oh_train_no_rif = pd.read_csv(os.path.join(one_hot_data_path, 'x_train.csv'))
    y_oh_train_no_rif = pd.read_csv(os.path.join(one_hot_data_path, 'y_train.csv'))
    x_oh_train, y_oh_train = retrain_modules.prepare_rif_setting(x_oh_train_no_rif, y_oh_train_no_rif)
    x_oh_train.to_csv(os.path.join(one_hot_data_path, 'x_train_rif.csv'), index=False)
    y_oh_train.to_csv(os.path.join(one_hot_data_path, 'y_train_rif.csv'), index=False)

y_oh_train = pd.Series(y_oh_train['InjurySeverity'].values)

# ------------------------------------------------------------------------------------------
st = time.time()
x_oh_rif_rus, y_oh_rif_rus = prepro_modules.data_resampling(x_oh_train, y_oh_train, sampling_method='RUS')
x_oh_rif_rus.to_csv(os.path.join(one_hot_data_path, 'x_train_rif_tl.csv'), index=False)
y_oh_rif_rus.to_csv(os.path.join(one_hot_data_path, 'y_train_rif_tl.csv'), index=False)
print("RUS finished on one_hot data with time", time.time() - st)

# st2 = time.time()
# x_noh_rif_rus, y_noh_rif_rus = prepro_modules.data_resampling(x_noh_train, y_noh_train, sampling_method='RUS')
# x_noh_rif_rus.to_csv(os.path.join(no_onehot_path, 'x_train_rif_rus.csv'), index=False)
# y_noh_rif_rus.to_csv(os.path.join(no_onehot_path, 'y_train_rif_rus.csv'), index=False)
# print("RUS finished on no_one_hot data with time", time.time() - st)

# TODO: TomekLinks cost over 7 days for each!