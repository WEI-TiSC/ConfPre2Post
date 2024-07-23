# -*- coding: utf-8 -*-
# Author : Junhao Wei
# @file : train_tab.py
# @Time : 2024/7/19 15:21
# Interpretation: 4. pretrain and retrain tab

import json
import os
import datetime

import joblib
import numpy as np
import optuna
import pandas as pd
import torch.optim
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight

from src.pkts.eval_metrics import f1_macro_for_eval
from src.pkts.loss_funcs import FocalLoss
from src.pkts.my_logger import logger
from src.pkts import preprocessing_modules as prepro_modules, eval_metrics
from src.train_flow import pre_processing


def objective_tab(feats, labels, trial, sampling='ROS', class_weights=None):
    if class_weights is None:
        class_weights = [1, 1, 1]
    feats = feats.drop(columns=['CASEWGT'])
    x_train, x_cv, y_train, y_cv = train_test_split(feats, labels, stratify=labels,
                                                    shuffle=True, test_size=0.2, random_state=42)

    if sampling != 'None':
        x_train, y_train = prepro_modules.data_resampling(x_train, y_train, sampling)
    else:
        classes = np.unique(y_train)
        class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)

    logger.info(f'class weight is: {class_weights}')
    logger.warning(f'Sampling: {sampling}, Data length: {len(x_train)} for training, '
                   f'{len(x_cv)} for Cross validation. '
                   f'Severe Positive ratio: '
                   f'{round(len(y_train[y_train == 2]) / len(y_train), 3)}, '
                   f'Slight Positive ratio: '
                   f'{round(len(y_train[y_train == 1]) / len(y_train), 3)}')

    # Hyper-tuning
    n_da = trial.suggest_int('n_d', 8, 32, step=4)  # na=nd
    n_steps = trial.suggest_int('n_steps', 1, 5)
    gamma = trial.suggest_float('gamma', 1.0, 2.0, step=0.05)
    n_shared = trial.suggest_int('n_shared', 1, 5)
    lambda_sparse = trial.suggest_float('lambda_sparse', 1e-6, 1e-3, log=True)
    n_independent = trial.suggest_int('n_independent', 1, 5)
    mask_type = trial.suggest_categorical('mask_type', ['sparsemax', 'entmax'])
    tab_params = dict(n_d=n_da, n_a=n_da, n_steps=n_steps, gamma=gamma,
                      lambda_sparse=lambda_sparse, n_independent=n_independent,
                      n_shared=n_shared, optimizer_fn=torch.optim.Adam,
                      mask_type=mask_type)

    scheduler_fn = torch.optim.lr_scheduler.ReduceLROnPlateau
    mode = 'max'  # f1-macro
    scheduler_params = {
        "mode": mode,  # max because of eval metric
        "factor": 0.5,  # Initial learning rate 0.01, lr decay = 0.5
        "patience": 10  # 10 epochs patience, when loss not get smaller over 10 epochs, lr_new = lr * factor
    }

    tab_clf = TabNetClassifier(scheduler_fn=scheduler_fn, scheduler_params=scheduler_params,
                               device_name='cuda', **tab_params)
    tab_clf.loss_fn = FocalLoss(alpha=torch.Tensor(class_weights))

    tab_clf.fit(
        X_train=x_train.values,
        y_train=y_train,
        eval_set=[(x_train.values, y_train), (x_cv.values, y_cv)],
        eval_name=['train', 'validation'],
        eval_metric=[f1_macro_for_eval],
        max_epochs=500,
        batch_size=256,
        patience=100,
        drop_last=False
    )

    y_pred = tab_clf.predict(x_cv.values)
    cv_score = f1_score(y_cv, y_pred, average='macro')
    logger.info(f'Current CV f1-macro score of is: {cv_score}')
    return cv_score


def hyperparam_tuning(feats, labels, sampling='ROS', n_trials=300):
    pruner = optuna.pruners.MedianPruner()
    study_info = 'TabNet_' + sampling
    study_direction = 'maximize'  # based on: metric == 'macro'
    print(f"Study aims to {study_direction} the metric: f1_macro!")
    study = optuna.create_study(direction=study_direction, study_name=study_info, pruner=pruner)
    study.optimize(lambda trial: objective_tab(feats, labels, trial, sampling=sampling), n_trials=n_trials)

    logger.debug(f'{study_info} get the Best f1_macro: {study.best_value:.3f}')

    save_result_pth = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                   f'trained_model_info/TabNet',
                                   f'{datetime.date.today()}_{sampling}_f1_macro')
    os.makedirs(save_result_pth, exist_ok=True)

    # 下面的添加存模型！
    with open(os.path.join(save_result_pth, 'param_dict.json'),
              encoding='utf-8', mode='w') as param_save:
        param_save.write(json.dumps(study.best_params, ensure_ascii=False, indent=4))

    with open(os.path.join(save_result_pth, 'feature_list.txt'),
              encoding='utf-8', mode='w') as feature_list:
        for feature in feats.columns.values:
            if feature == 'CASEWGT':
                continue
            feature_list.write(feature)
            feature_list.write('\n')

    return study.best_params, study.best_value, save_result_pth


def pre_train_tab(data_path, n_trials, sampling='ROS'):
    """
    If rif is wanted, please pre-process before this func!

    :param data_path: data path for all pre-processed ones
    :param n_trials: running trials for hyper-params tuning
    :param sampling: sampling used in pre-train (ROS)
    :return:
    """
    x_train = pd.read_csv(os.path.join(data_path, 'x_train.csv'))
    y_train = pd.read_csv(os.path.join(data_path, 'y_train.csv'))

    x_test = pd.read_csv(os.path.join(data_path, 'x_test.csv'))
    y_test = pd.read_csv(os.path.join(data_path, 'y_test.csv'))

    feats = pd.concat([x_train, x_test], axis=0).reset_index(drop=True)
    labels = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)
    labels = pd.Series(labels['InjurySeverity'].values)

    best_params, _, save_dir = hyperparam_tuning(feats, labels, sampling, n_trials)
    return save_dir


def retrain_tab_module(x_train,
                       x_test,
                       y_train,
                       y_test,
                       best_params,
                       sampling='None',
                       rif='no_rif',  # Only for saving path, pre-processing needs to be done before training
                       save_dir=None,
                       weights=None):
    """
    CASEWGT needs to be removed if no RIF!

    :param weights:
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :param best_params:
    :param sampling:
    :param rif:
    :param save_dir:
    :return:
    """
    weight_str = 'default'
    if weights is None:
        weights = [1, 1, 1]
    if sampling != 'None' or rif != 'no_rif':  # Use given class weights
        weight_str = '_'.join(str(val) for val in weights)
        if sampling != 'None':
            x_train, y_train = prepro_modules.data_resampling(x_train, y_train, sampling_method=sampling)
    else:
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)

    scheduler_fn = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = {
        "mode": 'min',
        "factor": 0.5,
        "patience": 1
    }

    tab_clf = TabNetClassifier(scheduler_fn=scheduler_fn, scheduler_params=scheduler_params,
                               device_name='cuda', **best_params)
    tab_clf.loss_fn = FocalLoss(alpha=torch.Tensor(weights))

    tab_clf.fit(
        X_train=x_train.values,
        y_train=y_train,
        eval_set=[(x_train.values, y_train), (x_test.values, y_test)],
        eval_name=['train', 'test'],
        eval_metric=[f1_macro_for_eval],
        # weights=1,  # 1 for automated balancing dict for custom weights per class
        max_epochs=1000,
        batch_size=512,
        patience=100,
        drop_last=False
    )

    y_pred = tab_clf.predict(x_test.values)
    print(metrics.classification_report(y_test, y_pred, digits=4))
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print('\nConfusion matrix:\n', conf_matrix)

    model_info = 'Retrain_Tab_' + rif + '_' + sampling + '_MultiClassification'
    save_path = os.path.join(save_dir, model_info, weight_str)
    os.makedirs(save_path, exist_ok=True)

    # Save evaluations
    eval_metrics.draw_confusion_matrix(y_test, y_pred, save_path, model_info)

    classification_report = metrics.classification_report(y_test, y_pred, digits=4, output_dict=True)
    classification_report = pd.DataFrame(classification_report).transpose()
    classification_report.to_csv(os.path.join(save_path, "classification_result.csv"), index=True)

    # Save class weights
    with open(os.path.join(save_path, 'class_weights.txt'), 'w') as f:
        f.write('{')
        for i, wg in enumerate(weights):
            f.write('\n')
            f.writelines(f'{i}:  {wg}')
        f.write('\n')
        f.write('}')

    # Save model as .pkl
    joblib.dump(tab_clf, os.path.join(save_path, f'{model_info}.pkl'))


def retrain_tab(data_path, param_path, weights):
    x_train = pd.read_csv(os.path.join(data_path, 'x_train.csv'))
    y_train = pd.read_csv(os.path.join(data_path, 'y_train.csv'))
    x_test = pd.read_csv(os.path.join(data_path, 'x_test.csv'))
    y_test = pd.read_csv(os.path.join(data_path, 'y_test.csv'))
    x_train_rif_rus = pd.read_csv(os.path.join(data_path, 'x_train_rif_rus.csv'))
    y_train_rif_rus = pd.read_csv(os.path.join(data_path, 'y_train_rif_rus.csv'))

    y_train = pd.Series(y_train['InjurySeverity'].values)
    y_train_rif_rus = pd.Series(y_train_rif_rus['InjurySeverity'].values)
    y_test = pd.Series(y_test['InjurySeverity'].values)

    if 'CASEWGT' in x_test.columns.values:
        x_test = x_test.drop(columns=['CASEWGT'])
    if 'CASEWGT' in x_train.columns.values:
        x_train = x_train.drop(columns=['CASEWGT'])

    with open(os.path.join(param_path, 'param_dict.json'), 'r') as f:
        best_params = json.load(f)

    rif_setting = [True, False]
    resampling_setting = ['None', 'ROS', 'ADASYN', 'SMOTETomek']

    if weights != [1, 1, 1]:
        del resampling_setting[0]

    for whether_rif in rif_setting:
        if whether_rif:
            retrain_tab_module(x_train_rif_rus, x_test, y_train_rif_rus, y_test, best_params,
                               rif='rif_rus', sampling='None', save_dir=param_path, weights=weights)
        else:
            for rsp in resampling_setting:
                retrain_tab_module(x_train, x_test, y_train, y_test, best_params,
                                   rif='no_rif', sampling=rsp, save_dir=param_path, weights=weights)


if __name__ == "__main__":
    DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                             'data', 'Combined', 'no_one_hot')  # Deep learning do not need one_hot
    # N_TRIALS = 100
    MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                              'trained_model_info', 'TabNet', '2024-07-23_ROS_f1_macro')
    CLASS_WEIGHTS = [[1, 1, 2], [1, 1, 3], [1, 1, 4], [1, 1, 5]]

    # save_path = pre_train_tab(DATA_PATH, N_TRIALS, sampling='ROS')
    for class_wgt in CLASS_WEIGHTS:
        retrain_tab(DATA_PATH, MODEL_PATH, class_wgt)
