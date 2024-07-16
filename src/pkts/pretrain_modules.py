# -*- coding: utf-8 -*-
# Author : Junhao Wei
# @file : pretrain_modules.py
# @Time : 2024/7/16 0:25
# Interpretation: This file is for pretrain functions, including parameter grids,
# training functions of LGBM, CB, RF, and also Optuna optimizations.

"""
Followings are parameter grids.
"""
import datetime
import json
import os
import warnings

import lightgbm
import numpy as np
import optuna
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import compute_class_weight

from src.pkts.my_logger import logger
from src.pkts.preprocessing_modules import data_resampling


def param_grid_dict_LGBM(trial):
    """
    For pre-train.

    LGBM params:
        1. 控制数结构的超参
            1.1. max_depth & num_leaves:树深度与叶子节点数，max_depth通常为3-8，
                num_leaves常为2&(max_depth);
            1.2. min_child_samples: 叶子节点向下分裂的最小样本数；大数据集通常设置为千级以上；

        2. 提高准确性的超参
            2.1. learning_rate & n_estimators: 学习率，梯度下降的步长参数；决策树数量，控制模型规格。
                常规而言，LGBM易于过拟合，而learning_rate一般设置在0.01-0.3之间。
                处理时，通常设置稍多子树（如1000）辅以较低的学习率，并通过early_stopping找到最优的迭代次数。
            2.2. max_bin (default=255): 变量分箱数，越多则信息保留越详细；反之则信息损失多。
                保存越详细则信息越具体，但对应会降低泛化能力！

        3. 控制过拟合的超参
            3.1. lambda_l1 & lambda_l2: L1 & L2正则化，搜索范围常在(0, 100);
            3.2. min_gain_to_split: 分裂的最小增益。保守搜索范围(0, 20);
            3.3. bagging_fraction & feature_fraction: 范围(0, 1);
                bagging_fraction: 每棵树训练时的训练样本百分比，和下面一样，用于让模型 好，而不同！
                feature_fraction: 每棵树训练时的采样特征百分比，避免每次分裂都用一个特征带来的同质化增强泛化性；
    """
    param_gird = {
        'n_estimators': trial.suggest_int('n_estimators', 20, 301, step=3),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.31, step=0.01),
        'num_leaves': trial.suggest_int('num_leaves', 10, 1001, step=10),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 2001, step=10),
        'reg_alpha': trial.suggest_int('reg_alpha', 0, 104, step=8),
        'reg_lambda': trial.suggest_int('reg_lambda', 0, 104, step=8),
        'min_split_gain': trial.suggest_float('min_split_gain', 0, 15, step=0.2),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.95, step=0.05),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.2, 0.95, step=0.05),
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart'])
    }

    return param_gird


def param_grid_dict_RF(trial):
    param_grid = {
        'n_estimators': trial.suggest_int('n_estimators', 20, 501, step=10),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 10, 1001, step=20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 1001, step=20),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
    }

    return param_grid


def param_grid_dict_CB(trial):
    param_grid = {
        'iterations': trial.suggest_categorical('iterations', [100, 200, 300, 500, 1000, 1200, 1500]),
        'learning_rate': trial.suggest_float("learning_rate", 0.001, 0.3, step=0.005),
        'random_strength': trial.suggest_int("random_strength", 1, 10),
        'bagging_temperature': trial.suggest_int("bagging_temperature", 0, 10),
        'max_bin': trial.suggest_categorical('max_bin', [4, 5, 6, 8, 10, 20, 30]),
        'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
        'min_data_in_leaf': trial.suggest_int("min_data_in_leaf", 1, 10),
        'od_type': "Iter",
        'od_wait': 100,
        "depth": trial.suggest_int("depth", 2, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 100, log=True),
        'one_hot_max_size': trial.suggest_categorical('one_hot_max_size', [5, 10, 12, 100, 500, 1024]),
    }

    return param_grid


def param_grid_dict(trial, model):
    """
    Find corresponding param grim dict.

    :param trial: param selector.
    :param model: model used.
    :return: params.
    """
    if model == 'LGBM':
        return param_grid_dict_LGBM(trial)
    elif model == 'RF':
        return param_grid_dict_RF(trial)
    elif model == 'CB':
        return param_grid_dict_CB(trial)


def pre_train_LGBM(x_train, x_cv, y_train, y_cv, param_grids, cv_scores, idx, metric='f1-macro', class_weight=None):
    """
    Pretrain LGBM for single round.

    :param x_train:
    :param x_cv:
    :param y_train:
    :param y_cv:
    :param param_grids:
    :param metric:
    :param cv_scores:
    :param idx:
    :param class_weight:
    :return:
    """
    model = lightgbm.LGBMClassifier(objective='multiclass', verbosity=-1, class_weight=class_weight, **param_grids)

    model.fit(
        x_train, y_train,
        # eval_set=[(x_cv, y_cv)],
        # eval_metric=lambda y_true, y_pred: [f1_score(y_true, y_pred, average='weighted')],
        # callbacks=[lightgbm.early_stopping(150, verbose=False)]
    )

    pred_y = model.predict(x_cv)

    if metric == 'auc':
        cv_scores[idx] = roc_auc_score(y_cv, pred_y, average='macro', multi_class='ovr')
    else:
        if metric == 'f1-macro':
            average = 'macro'
        else:
            average = 'weighted'

        cv_scores[idx] = f1_score(y_cv, pred_y, average=average)


def pre_train_RF(x_train, x_cv, y_train, y_cv, param_grids, cv_scores, idx, metric='f1-macro', class_weight=None):
    """
    Pretrain RF for single round.


    :param x_train:
    :param x_cv:
    :param y_train:
    :param y_cv:
    :param param_grids:
    :param cv_scores:
    :param idx:
    :return:
    """
    model = RandomForestClassifier(oob_score=True, random_state=42, class_weight=class_weight, **param_grids)
    model.fit(x_train, y_train)
    pred_y = model.predict(x_cv)

    if metric == 'auc':
        cv_scores[idx] = roc_auc_score(y_cv, pred_y, average='macro', multi_class='ovr')
    else:
        if metric == 'f1-macro':
            average = 'macro'
        else:
            average = 'weighted'

        cv_scores[idx] = f1_score(y_cv, pred_y, average=average)


def pre_train_CB(x_train, x_cv, y_train, y_cv, param_grids, cv_scores, idx, metric='f1-macro', class_weight=None):
    """
    Pretrain CB for single round.

    :param class_weight:
    :param metric:
    :param x_train:
    :param x_cv:
    :param y_train:
    :param y_cv:
    :param param_grids:
    :param cv_scores:
    :param idx:
    :return:
    """
    model = CatBoostClassifier(verbose=False, loss_function='MultiClassOneVsAll',
                               class_weights=class_weight, **param_grids)
    model.fit(x_train, y_train, early_stopping_rounds=150)
    pred_y = model.predict(x_cv)

    if metric == 'auc':
        cv_scores[idx] = roc_auc_score(y_cv, pred_y, average='macro', multi_class='ovr')
    else:
        if metric == 'f1-macro':
            average = 'macro'
        else:
            average = 'weighted'

        cv_scores[idx] = f1_score(y_cv, pred_y, average=average)


def objective(feats, labels, param_grid, trial, metric='f1-macro', use_model='LGBM', k_fold=6,
              sampling='None', given_class_weights=None):
    """
    Pre-train. 使用 StratifiedKFold 分层采样确保各个集合中各类别样本比例与原始数据集相同。
    由于后续会进行retrain，到时再分割训练测试即可，所以当前直接全部数据一起丢进来训练就好！

    进入当前轮交叉验证 -> 分割CV -> 训练集重采样(包括RIF) -> 循环调参
    * RIF不需要k折cv故另起函数

    :param given_class_weights: 预先指定的class weight
    :param metric: 指标
    :param trial: 轮数
    :param labels: 非测试集的label
    :param feats: 非测试机的feature
    :param param_grid: 参数池
    :param use_model: 使用的模型
    :param k_fold: 交叉验证的折数
    :param sampling: 是否对训练集重采样
        None: 不重采样；
        'ADASYN': adasyn重采样 （ 暂时考量仅用于无RIF数据集！ ）
        'RUS': RandomUnderSampling （ 对于RIF数据 ）
        'ARF': Adversial Random Forests重采样 （ 暂时考量仅用于无RIF数据集！ )
        etc...
    :return:
    """
    stratified_k_fold = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)
    cv_scores = np.empty(k_fold)

    # 无RIF故CASEWGT直接删去
    feats = feats.drop(columns=['CASEWGT'])

    for idx, (train_idx, cv_idx) in enumerate(stratified_k_fold.split(feats, labels)):
        x_train, x_cv = feats.iloc[train_idx], feats.iloc[cv_idx]
        y_train, y_cv = labels.iloc[train_idx], labels.iloc[cv_idx]
        # 处理重采样
        if sampling != 'None':
            x_train, y_train = data_resampling(x_train, y_train, sampling_method=sampling)
            class_weights = given_class_weights
        else:
            classes = np.unique(y_train)
            weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
            class_weights = dict(zip(classes, weights))

        if idx == 0:
            print(f'Class weight is: {class_weights}')
            logger.warning(f'Sampling: {sampling}, Data length: {len(x_train)} for training, '
                           f'{len(x_cv)} for Cross validation. '
                           f'Severe Positive ratio: '
                           f'{round(len(y_train[y_train == 2]) / len(y_train), 3)}, '
                           f'Slight Positive ratio: '
                           f'{round(len(y_train[y_train == 1]) / len(y_train), 3)}')
        if use_model == 'LGBM':
            pre_train_LGBM(x_train, x_cv, y_train, y_cv, param_grid, cv_scores, idx, metric, class_weights)
        elif use_model == 'RF':
            pre_train_RF(x_train, x_cv, y_train, y_cv, param_grid, cv_scores, idx, metric, class_weights)
        elif use_model == 'CB':
            pre_train_CB(x_train, x_cv, y_train, y_cv, param_grid, cv_scores, idx, metric, class_weights)
        else:
            raise ValueError("Unknown ML Model!")

    return np.mean(cv_scores)


def param_search_by_optuna(feats, labels, sampling, study_name='LGBM', metric='f1-macro', class_weights=None,
                           n_trials=300, one_hot=False):
    """
    pre-train, 调用函数寻找最佳参数！


    :param class_weights: inherited class weight
    :param metric: Evaluation metric
    :param n_trials: running trials
    :param sampling: sampling tech
    :param study_name: model name
    :param feats: features
    :param labels: labels
    :param one_hot: Whether one-hot
    :return: (study.best_params, study.best_value, result_dir)
    """
    assert study_name in ['LGBM', 'RF', 'CB']  # ML here
    warnings.filterwarnings("ignore", category=UserWarning)

    pruner = optuna.pruners.MedianPruner()
    study_info = study_name + '_' + sampling if sampling != 'None' else study_name + '_NoSampling'
    study = optuna.create_study(direction='maximize', study_name=study_info, pruner=pruner)

    func = lambda trial: objective(feats=feats, labels=labels, param_grid=param_grid_dict(trial, study_name),
                                   trial=trial,
                                   metric=metric, use_model=study_name, sampling=sampling,
                                   given_class_weights=class_weights)
    study.optimize(func, n_trials=n_trials)

    logger.debug(f'{study_info}_{metric} get the Best {metric}: {study.best_value:.3f}')

    # Set save dir
    one_hot_str = 'NoOnehot' if not one_hot else 'Onehot'
    result_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                              f'trained_model_info/{study_name}',
                              f'{datetime.date.today()}_{sampling}_{one_hot_str}_{metric}')
    os.makedirs(result_dir, exist_ok=True)
    pre_train_result_dir = os.path.join(result_dir, 'pre_train_info')
    os.makedirs(pre_train_result_dir, exist_ok=True)

    if n_trials > 150:  # small trials for debug
        fig = optuna.visualization.plot_optimization_history(study)  # 要用plotly的离线打印
        # plotly.offline.plot(fig)
        fig.write_html(os.path.join(pre_train_result_dir,
                                    f'optimization history of {study_name}_trial_{n_trials}.html'))

        # Save hyper-param results
        with open(os.path.join(pre_train_result_dir, 'param_dict.json'),
                  encoding='utf-8', mode='w') as param_save:
            param_save.write(json.dumps(study.best_params, ensure_ascii=False, indent=4))
        # Save used features.
        with open(os.path.join(pre_train_result_dir, 'feature_list.txt'),
                  encoding='utf-8', mode='w') as feature_list:
            for feature in feats.columns.values:
                if feature == 'CASEWGT':
                    continue
                feature_list.write(feature)
                feature_list.write('\n')

    return study.best_params, study.best_value, result_dir




