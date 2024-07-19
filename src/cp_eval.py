# -*- coding: utf-8 -*-
# Author : Junhao Wei
# @file : cp_eval.py
# @Time : 2024/7/19 18:02
# Interpretation
import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.conformal_algo import naive_CP, APS_CP, CCCP, eval_CP
from src.pkts.my_logger import logger


def eval_with_cp(model_name, model_path, x_train, y_train, x_test, y_test,
                 pure_train=False, cp_method='Naive', alpha=0.1):
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
    :param pure_train: Whether 1/2 test data is used for calibration
    :return:
    """
    clf = joblib.load(model_path)

    if not pure_train:
        x_test, x_half_calib, y_test, y_half_calib = train_test_split(x_test, y_test, stratify=y_test,
                                                                      shuffle=True, test_size=0.5, random_state=42)
        x_calib = pd.concat([x_train, x_half_calib]).reset_index(drop=True)
        y_calib = pd.concat([y_train, y_half_calib]).reset_index(drop=True)
    else:
        x_calib = x_train
        y_calib = y_train

    if model_name == 'TN':
        y_calib_softmax = clf.predict_proba(x_calib.values)
        y_test_softmax = clf.predict_proba(x_calib.values)
    else:
        y_calib_softmax = clf.predict_proba(x_calib)
        y_test_softmax = clf.predict_proba(x_calib)

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
    save_path = os.path.join(os.path.dirname(os.path.dirname(model_path)), f'{model_info}')
    os.makedirs(save_path, exist_ok=True)

    eval_CP.draw_coverage(conf_sets, y_test, cp_method, save_path, model_info, alpha=alpha)
    eval_CP.draw_set_sizes(conf_sets, cp_method, save_path, model_info, alpha=alpha)

    # TODO: save CP results for hard sample analysis
