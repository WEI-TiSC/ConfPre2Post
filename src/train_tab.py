# -*- coding: utf-8 -*-
# Author : Junhao Wei
# @file : train_tab.py
# @Time : 2024/7/19 15:21
# Interpretation: Implement of TabNet

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


def objective_tab(feats, labels, trial, sampling='None', class_weights={0: 1, 1: 1, 2: 6}):
    if class_weights is None:
        class_weights = {0: 1, 1: 1, 2: 6}
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
        "factor": 0.1, # Initial learning rate 0.01, lr decay = 0.1
        "patience": 10 # 10 epochs patience, when loss not get smaller over 10 epochs, lr_new = lr * factor
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
        max_epochs=300,
        batch_size=256,
        patience=60,
        drop_last=False
    )

    y_pred = tab_clf.predict(x_cv.values)
    cv_score = f1_score(y_cv, y_pred, average='macro')
    logger.info(f'Current CV f1-macro score of is: {cv_score}')
    return cv_score


def hyperparam_tuning(feats, labels, sampling='None', n_trials=300):
    pruner = optuna.pruners.MedianPruner()
    study_info = 'TabNet_' + sampling
    study_direction = 'maximize'  # based on: metric == 'macro'
    print(f"Study aims to {study_direction} the metric: f1_macro!")
    study = optuna.create_study(direction=study_direction, study_name=study_info, pruner=pruner)
    study.optimize(lambda trial: objective_tab(feats, labels, trial, sampling=sampling), n_trials=n_trials)

    logger.debug(f'{study_info} get the Best f1_macro: {study.best_value:.3f}')

    save_result_pth = os.path.join(os.path.dirname(os.path.dirname(__file__)), f'trained_model_info/TabNet',
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


def fine_tune(x_train,
              x_test,
              y_train,
              y_test,
              best_params,
              sampling='None',
              rif='no_rif',  # Only for saving path, pre-processing needs to be done before training
              save_dir=None,
              weights={0: 1, 1: 1, 2: 8}):
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
    if sampling != 'None':
        x_train, y_train = prepro_modules.data_resampling(x_train, y_train, sampling_method=sampling)
    else:
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)

    scheduler_fn = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = {
        "mode": 'min',
        "factor": 0.1,
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
        max_epochs=3000,
        batch_size=512,
        patience=100,
        drop_last=False
    )

    y_pred = tab_clf.predict(x_test.values)
    print(metrics.classification_report(y_test, y_pred, digits=4))
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print('\nConfusion matrix:\n', conf_matrix)

    model_info = 'Retrain_Tab_' + rif + '_' + sampling + '_MultiClassification'
    save_path = os.path.join(save_dir, model_info)
    os.makedirs(save_path, exist_ok=True)

    # Save evaluations
    eval_metrics.draw_confusion_matrix(y_test, y_pred, save_path, model_info)

    classification_report = metrics.classification_report(y_test, y_pred, digits=4, output_dict=True)
    classification_report = pd.DataFrame(classification_report).transpose()
    classification_report.to_csv(os.path.join(save_path, "classification_result.csv"), index=True)

    # Save model as .pkl
    joblib.dump(tab_clf, os.path.join(save_path, f'{model_info}.pkl'))


def wrap_up_learning(feats, labels, x_train, x_test, y_train, y_test, n_trials, sampling='None', rif='no_rif'):
    '''
    If rif is wanted, please pre-process before this func!

    :param feats:
    :param labels:
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :param n_trials:
    :param sampling:
    :param rif:
    :return:
    '''
    best_params, _, save_dir = hyperparam_tuning(feats, labels, sampling, n_trials)

    fine_tune(x_train, x_test, y_train, y_test, best_params, sampling, rif, save_dir)


if __name__ == "__main__":
    DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             'data', 'Combined', 'CombineNassCiss.csv')
    USE_FEATS = prepro_modules.ALL_PRE_FEATURES
    WHETHER_RIF = 'no_rif'
    SAMPLING = 'None'
    N_TRIALS = 200

    feats, labels, x_train, x_test, y_train, y_test = (
        pre_processing.get_processed_data(DATA_PATH, use_features=USE_FEATS))

    # Input rif-ed data here if rif is wanted
    if WHETHER_RIF != 'no_rif':
        rif_data_path = 'Here to fill your path'
        x_train = pd.read_csv(os.path.join(rif_data_path, 'x_train_rif.csv'))
        y_train = pd.read_csv(os.path.join(rif_data_path, 'y_train_rif.csv'))
    else:
        x_train = x_train.drop(columns=['CASEWGT'])
    x_test = x_test.drop(columns=['CASEWGT'])

    wrap_up_learning(feats, labels, x_train, x_test, y_train, y_test, N_TRIALS, SAMPLING, rif=WHETHER_RIF)

# TODO: Try f score beta (see firefox)