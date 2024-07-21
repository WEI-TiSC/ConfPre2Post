# -*- coding: utf-8 -*-
# Author : Junhao Wei
# @file : train_ml_models.py
# @Time : 2024/7/16 1:12
# Interpretation:
# this file completes pre-train -> retrain procedure for specific (model, sampling, one_hot).
from src.pkts.pretrain_modules import param_search_by_optuna
from src.pkts import retrain_modules


def train_single_model_sampling(feats, labels,
                                x_train, x_test, y_train, y_test,
                                model, trials, one_hot=False,
                                metric='f1-macro', class_weights=None):
    """
    Pretrain single model based on specific sampling and one-hot choice.

    :param rif: Whether rif
    :param feats: for pretrain, x
    :param labels: for pretrain, y
    :param x_train: for retrain, x
    :param x_test: for evaluation, x
    :param y_train: for retrain, y
    :param y_test: for evaluation, y
    :param model: MODEL to train
    :param trials: running trials for hyper-param searching
    :param one_hot: Whether one-hot used, default false
    :param metric: evaluation metric, default f1-macro
    :param class_weights: inherited class weights, default None
    :return: training results (for evaluating model performance!)
    """
    # Pretrain for hyper-params tuning
    best_params, _, result_dir = param_search_by_optuna(feats, labels, study_name=model,
                                                        metric=metric, class_weights=class_weights,
                                                        n_trials=trials, one_hot=one_hot)

    # Retrain for performance evaluation
    # rif_str = 'with_rif' if rif else 'no_rif'
    # un_samp = 'None'
    # if rif:
    #     un_samp = 'TomekLinks'  # add: Never use None, use rus and tl! (data prepared)
    #     x_train, y_train, x_test = retrain_modules.prepare_rif_setting(x_train, y_train, x_test)
    # use_sampling = un_samp if un_samp != 'None' else sampling
    # retrain_modules.retrain(model, best_params, x_train, y_train, x_test, y_test, sampling=use_sampling,
    #                         class_weights=class_weights, save_dir=result_dir, rifed=rif_str)

# TODO: Modify retrain to learn all sampling.
