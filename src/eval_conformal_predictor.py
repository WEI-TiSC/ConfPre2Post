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
from src.pkts.preprocessing_modules import ALL_PRE_FEATURES


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

    # Change conf_set
    conf_softmax = [apply_softmax_to_dict(cf_st) for cf_st in conf_sets]
    conf_float = [tensor_to_float_dict(cf_sfmx) for cf_sfmx in conf_softmax]
    conf_series = pd.Series(conf_float)

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
            f.write('Average set size of class %s is: %s \n' % (i, sz))

    return conf_series


def eval_with_cp_and_save_conf_set(model_name, model_path, x_calib, y_calib, x_test, y_test,
                 x_anl, cp_method='Naive', alpha=0.1):
    """
    Eval with CP and save conf set

    :param alpha:
    :param cp_method:
    :param model_name:
    :param model_path:
    :param x_calib:
    :param y_calib:
    :param x_test:
    :param y_test:
    :return:
    """
    clf = joblib.load(model_path)


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

    # Change conf_set
    conf_softmax = [apply_softmax_to_dict(cf_st) for cf_st in conf_sets]
    conf_float = [tensor_to_float_dict(cf_sfmx) for cf_sfmx in conf_softmax]
    conf_series = pd.Series(conf_float, name='Conformal Set')

    set_size = conf_series.apply(len)
    set_size.name = 'Set Size'
    analysis_data = pd.concat([x_anl, y_test, conf_series, set_size], axis=1)

    # Evaluation
    model_info = model_name + '_' + cp_method
    save_path = os.path.join(os.path.dirname(model_path), f'{model_info}')
    os.makedirs(save_path, exist_ok=True)

    analysis_data.to_csv(os.path.join(os.path.dirname(save_path), 'analysis_data.csv'), index=False)  # TODO: find covered cases in Jupyter!

    eval_CP.draw_coverage(conf_sets, y_test, cp_method, save_path, model_info, alpha=alpha)
    eval_CP.draw_set_sizes(conf_sets, cp_method, save_path, model_info, alpha=alpha)

    avg_size, avg_size_by_class = eval_CP.calc_avg_set_size_by_class(conf_sets, y_test)
    ssc_metric = eval_CP.calc_ssc_metric(conf_sets, y_test)

    with open(os.path.join(save_path, 'Eval_info.txt'), 'w') as f:
        f.write('SSC Metric is: %s \n' % ssc_metric)
        f.write('Total average set size is: %s \n' % avg_size)

        for i, sz in enumerate(avg_size_by_class):
            f.write('Average set size of class %s is: %s \n' % (i, sz))



def apply_softmax_to_dict(conf_set):
    """
    Change tensor to softmax.

    :param conf_set: single conf_set
    :return:
    """
    keys = list(conf_set.keys())
    vals = torch.tensor([conf_set[key].item() for key in keys])
    total_sum = vals.sum()
    normalized_vals = vals/total_sum
    return {keys[i]: normalized_vals[i].unsqueeze(0) for i in range(len(keys))}


def tensor_to_float_dict(tensor_dict):
    return {key: round(val.item(), 4) for key, val in tensor_dict.items()}


if __name__ == '__main__':
    model_name = 'TabNet'
    pretrain_info = '2024-07-23_ROS_f1_macro'
    retrain_info = 'Retrain_Tab_no_rif_ROS_MultiClassification'
    class_weight_info = '1_1_3'
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              'trained_model_info', model_name, pretrain_info, retrain_info,
                              class_weight_info, f'{retrain_info}.pkl')
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'Combined',
                             'CombineNassCiss.csv')

    use_feats = ALL_PRE_FEATURES
    use_feats.remove('CASEWGT')
    use_feats.remove('InjurySeverity')

    df_raw = pd.read_csv(data_path, index_col=0)  # raw data, processing required
    for feat in use_feats:
        df_raw = df_raw[df_raw[feat] != 65536]  # del null for all features

    feats, labels = df_raw.drop(columns=['InjurySeverity']), df_raw['InjurySeverity']
    x_train, x_test_calib_anl, y_train, y_test_calib_anl = train_test_split(feats, labels, stratify=labels,
                                                                shuffle=True, test_size=0.20,
                                                                random_state=42)
    x_train = x_train[use_feats]

    # Calib get here
    x_test_anl, x_half_calib, y_test, y_half_calib = train_test_split(x_test_calib_anl, y_test_calib_anl,
                                                                          stratify=y_test_calib_anl, shuffle=True,
                                                                          test_size=0.5, random_state=42)
    x_half_calib = x_half_calib[use_feats]
    x_test = x_test_anl[use_feats]  # Mapping! Concat other results with x_test_anl later.

    x_test_anl = x_test_anl.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    x_calib = pd.concat([x_train, x_half_calib], axis=0).reset_index(drop=True)
    y_calib = pd.concat([y_train, y_half_calib], axis=0).reset_index(drop=True)

    eval_with_cp_and_save_conf_set(model_name, model_path, x_calib, y_calib, x_test, y_test, x_test_anl,
                                cp_method='CCCP', alpha=0.10)
