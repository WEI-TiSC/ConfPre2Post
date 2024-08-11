# -*- coding: utf-8 -*-
# Author : Junhao Wei
# @file : eval_conformal_predictor.py
# @Time : 2024/7/19 18:02
# Interpretation
import os

import joblib
import pandas as pd
import torch.nn
from sklearn.model_selection import train_test_split

from src.conformal_algo import naive_CP, APS_CP, CCCP, eval_CP
from src.pkts.my_logger import logger


def eval_with_cp(model_name, model_path, x_train, y_train, x_test, y_test,
                 test_in_calib=False, cp_method='Naive', alpha=0.1):
    """
    Retrain with CP or Eval with CP.
        Retrain: Data not that sufficient but
        Eval: Get half of test data out, as (train_data, 1/2 test data) to do simple calibration?

    :param alpha:
    :param cp_method:
    :param model_name:
    :param model_path:
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param test_in_calib: Whether 1/2 test data is used for calibration
    :return:
    """
    clf = joblib.load(model_path)

    if test_in_calib:
        x_test, x_half_calib, y_test, y_half_calib = train_test_split(x_test, y_test, stratify=y_test,
                                                                      shuffle=True, test_size=0.5, random_state=42)
        x_calib = pd.concat([x_train, x_half_calib], axis=0).reset_index(drop=True)
        y_calib = pd.concat([y_train, y_half_calib], axis=0).reset_index(drop=True)
    else:
        x_calib = x_train
        y_calib = y_train

    if model_name == 'TabNet':
        y_calib_prob = clf.predict_proba(x_calib.values)
        y_test_prob = clf.predict_proba(x_test.values)
    else:
        y_calib_prob = clf.predict_proba(x_calib)
        y_test_prob = clf.predict_proba(x_test)

    y_calib_tensor = torch.tensor(y_calib_prob)
    y_test_tensor = torch.tensor(y_test_prob)

    # 初始化 Softmax 对象并在指定维度上应用
    softmax = torch.nn.Softmax(dim=1)
    y_calib_softmax = softmax(y_calib_tensor)
    y_test_softmax = softmax(y_test_tensor)

    # Calibration
    if cp_method == 'Naive':
        q_hat = naive_CP.calc_qhat_naive_classification(y_calib_softmax, y_calib, alpha=alpha)
        logger.info('q_hat is % s' % q_hat)
        conf_sets = naive_CP.get_confsets_naive(y_test_softmax, q_hat)

    elif cp_method == 'APS':
        q_hat = APS_CP.calc_qhat_aps(y_calib_softmax, y_calib, alpha=alpha)
        logger.info('q_hat is % s' % q_hat)
        conf_sets = APS_CP.get_confsets_APS(y_test_softmax, q_hat)

    else:  # cp = CCCP
        q_hat_dict = CCCP.calc_qhat_CCCP(y_calib_softmax, y_calib, alpha=alpha)
        logger.info('q_hat_dict is % s' % q_hat_dict)
        conf_sets = CCCP.get_confsets_CCCP(y_test_softmax, q_hat_dict)

    # Evaluation
    model_info = model_name + cp_method
    save_path = os.path.join(os.path.dirname(model_path), f'{model_info}')
    os.makedirs(save_path, exist_ok=True)

    eval_CP.draw_coverage(conf_sets, y_test, cp_method, save_path, model_info, alpha=alpha)
    eval_CP.draw_set_sizes(conf_sets, cp_method, save_path, model_info, alpha=alpha)

    avg_size, avg_size_by_class = eval_CP.calc_avg_set_size_by_class(conf_sets, y_test)
    ssc_metric = eval_CP.calc_ssc_metric(conf_sets, y_test)

    with open(os.path.join(save_path, 'Eval_info.txt'), 'w') as f:
        f.write('SSC Metric is: %s \n' % ssc_metric)
        f.write('Total average set size is: %s \n' % avg_size)

        for i, sz in enumerate(avg_size_by_class):
            f.write('Average set size of class %s is: %s \n' %(i, sz))

    return y_test_softmax


if __name__ == '__main__':
    model_name = 'RF'
    pretrain_info = '2024-07-24_Onehot_f1-macro'
    retrain_info = 'retrain_RF_ROS_no_rif_MultiClassification'
    class_weight_info = '1_1_3'
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              'trained_model_info', model_name, pretrain_info, retrain_info,
                              class_weight_info, f'{retrain_info}.pkl')
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'Combined',
                             'with_one_hot')
    x_train = pd.read_csv(os.path.join(data_path, 'x_train.csv'))
    y_train = pd.read_csv(os.path.join(data_path, 'y_train.csv'))
    x_test = pd.read_csv(os.path.join(data_path, 'x_test.csv'))
    y_test = pd.read_csv(os.path.join(data_path, 'y_test.csv'))
    y_train = pd.Series(y_train['InjurySeverity'].values)
    y_test = pd.Series(y_test['InjurySeverity'].values)

    if 'CASEWGT' in x_train.columns:
        x_train = x_train.drop(columns=['CASEWGT'])
        x_test = x_test.drop(columns=['CASEWGT'])

    eval_with_cp(model_name, model_path, x_train, y_train, x_test, y_test,
                 test_in_calib=False, cp_method='Naive', alpha=0.1)
