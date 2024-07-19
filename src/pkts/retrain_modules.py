# -*- coding: utf-8 -*-
# Author : Junhao Wei
# @file : retrain_modules.py
# @Time : 2024/7/16 1:27
# Interpretation
import os

import joblib
import lightgbm
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import compute_class_weight

from src.pkts import preprocessing_modules as prepro_modules, eval_metrics
from src.pkts.my_logger import logger


def prepare_rif_setting(x_train, y_train):
    x_train_rif, y_train_rif = prepro_modules.run_rif(x_train, y_train)

    return x_train_rif, y_train_rif


def retrain(model, params, x_train, y_train, x_test, y_test, sampling='None', rifed='no_rif',
            class_weights=None, save_dir=None):
    """
    Retrain model.

    :param rifed: whether rif, only used for saving path
    :param model: model used
    :param params: hyper-params from pre-train
    :param x_train: x_train for training
    :param y_train: y_train for training
    :param x_test: x_test for evaluation
    :param y_test: y_test for evaluation
    :param sampling: sampling tech
    :param class_weights: inherited class weights
    :param save_dir: path for saving result
    :return:
    """
    assert model in ['LGBM', 'CB', 'RF'], 'Unknown model!'

    if sampling != 'None':  # Use given class weights
        x_train, y_train = prepro_modules.data_resampling(x_train, y_train, sampling_method=sampling)
    else:  # Calc class weights
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weights = dict(zip(classes, weights))

    # Prepare model
    if model == 'LGBM':
        retrain_model = lightgbm.LGBMClassifier(objective='multiclass', verbosity=-1, class_weight=class_weights,
                                                **params)
    elif model == 'RF':
        retrain_model = RandomForestClassifier(oob_score=True, random_state=42, class_weight=class_weights, **params)
    elif model == 'CB':
        retrain_model = CatBoostClassifier(verbose=True, loss_function='MultiClassOneVsAll',
                                           class_weights=class_weights, **params)
    else:
        raise ValueError("Unknown model!")

    logger.warning(f'Sampling: {sampling}, Data length: {len(x_train)} for training, '
                   f'{len(x_test)} for test. '
                   f'Severe Positive ratio in train: '
                   f'{round(len(y_train[y_train == 2]) / len(y_train), 3)}'
                   f' and Severe Positive ratio in test: '
                   f'{round(len(y_test[y_test == 2]) / len(y_test), 3)}')

    retrain_model.fit(x_train, y_train)

    # Predict & evaluation
    y_pred = retrain_model.predict(x_test)
    y_proba = retrain_model.predict_proba(x_test)
    print(metrics.classification_report(y_test, y_pred, digits=4))
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print('\n' * 3, 'Confusion matrix:\n', conf_matrix)

    # Draw Heatmap
    model_info = 'retrain_' + model + '_' + rifed + '_MultiClassification'
    model_save_dir = os.path.join(save_dir, model_info)
    os.makedirs(model_save_dir, exist_ok=True)

    eval_metrics.draw_confusion_matrix(y_test, y_pred, model_save_dir, model_info)
    # eval_metrics.draw_roc_curve(y_test, y_proba, model_save_dir, model_info)

    classification_report = metrics.classification_report(y_test, y_pred, digits=4, output_dict=True)
    classification_report = pd.DataFrame(classification_report).transpose()
    classification_report.to_csv(os.path.join(model_save_dir, "classification_result.csv"), index=True)

    # Save model as .pkl
    joblib.dump(retrain_model, os.path.join(model_save_dir, f'{model_info}.pkl'))
