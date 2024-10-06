# -*- coding: utf-8 -*-
# Author : Junhao Wei
# @file : eval_classification.py
# @Time : 2024/7/24 1:49
# Interpretation: 5. re-calc eval metrics using existing model & test data.
import copy
import os
import re

import joblib
import pandas as pd
from sklearn import metrics, logger
from sklearn.metrics import f1_score, recall_score


def recalc_classification_metrics(model_path, data_path, one_hot):
    """
    Calc numeric evaluation for multi-class and simplified binary problem.
    :param model_path: path of *.pkl
    :param data_path: path of features, labels for eval
    :param one_hot: Whether feats are with one-hot encoding
    :return: None
    """
    model = joblib.load(model_path)
    eval_path = os.path.join(os.path.dirname(model_path), 'numeric_evaluation')
    os.makedirs(eval_path, exist_ok=True)

    if one_hot:
        feats = pd.read_csv(os.path.join(data_path, 'with_one_hot', 'x_test.csv'))
        labels = pd.read_csv(os.path.join(data_path, 'with_one_hot', 'y_test.csv'))
    else:
        feats = pd.read_csv(os.path.join(data_path, 'no_one_hot', 'x_test.csv'))
        labels = pd.read_csv(os.path.join(data_path, 'no_one_hot', 'y_test.csv'))

    feats = feats.drop(columns=['CASEWGT'])

    if 'TabNet' in model_path:
        preds = model.predict(feats.values)
    else:
        preds = model.predict(feats)
    conf_matrix = metrics.confusion_matrix(labels, preds)
    conf_matrix_df = pd.DataFrame(conf_matrix, index=[0, 1, 2], columns=[0, 1, 2])
    conf_matrix_df.to_csv(os.path.join(eval_path, 'confusion_matrix.csv'))
    f1_weighted = round(f1_score(labels, preds, average='weighted'), 4)
    f1_macro = round(f1_score(labels, preds, average='macro'), 4)

    # Change into binary and calc bin
    bin_labels = copy.deepcopy(labels)
    bin_labels[bin_labels == 1] = 0  # Change all slight as No injury
    bin_labels[bin_labels == 2] = 1  # Severe as 1

    bin_preds = copy.deepcopy(preds)
    bin_preds[bin_preds == 1] = 0
    bin_preds[bin_preds == 2] = 1

    bin_conf_matrix = metrics.confusion_matrix(bin_labels, bin_preds)
    bin_conf_matrix_df = pd.DataFrame(bin_conf_matrix, index=[0, 1], columns=[0, 1])
    bin_conf_matrix_df.to_csv(os.path.join(eval_path, 'bin_confusion_matrix.csv'))
    bin_f1_weighted = round(f1_score(bin_labels, bin_preds, average='weighted'), 4)
    bin_f1_macro = round(f1_score(bin_labels, bin_preds, average='macro'), 4)
    bin_recall = round(recall_score(bin_labels, bin_preds), 4)

    # Save as dict
    save_form = {
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'bin_f1_weighted': bin_f1_weighted,
        'bin_f1_macro': bin_f1_macro,
        'bin_severe_recall': bin_recall
    }

    with open(os.path.join(eval_path, 'eval_classification.txt'), 'w', encoding='utf-8') as f:
        f.write('{')
        for eval_name, score in save_form.items():
            f.write('\n')
            f.writelines(f'{eval_name}:  {score}')
        f.write('\n')
        f.write('}')


if __name__ == '__main__':
    model_list_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                  'trained_model_info')
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                             'data', 'Combined')
    model_names = os.listdir(model_list_dir)
    for model_name in model_names:
        if model_name != 'LGBM':
            continue
        cur_model_dir = os.path.join(model_list_dir, model_name)
        model_settings = os.listdir(cur_model_dir)
        for model_setting in model_settings:  # Whether onehot here
            if not model_setting.startswith('2024-07-30'):
                continue
            one_hot = False
            if '_Onehot' in model_setting:
                one_hot = True
            model_setting_dir = os.path.join(cur_model_dir, model_setting)
            model_info_paths = os.listdir(model_setting_dir)
            # Find model *.pkl
            for model_info in model_info_paths:
                if model_info.startswith('pre'):
                    continue
                class_weight_path = os.path.join(model_setting_dir, model_info)
                class_weight_settings = os.listdir(class_weight_path)
                for weight_setting in class_weight_settings:
                    trained_model_path = os.path.join(class_weight_path, weight_setting)
                    for trained_info in os.listdir(trained_model_path):
                        if trained_info.endswith('.pkl'):
                            logger.info(f'Find modelin {trained_model_path} as {trained_info}, prepare to eval...')
                            abs_model_path = os.path.join(trained_model_path, trained_info)
                            recalc_classification_metrics(model_path=abs_model_path,
                                                          data_path=data_path,
                                                          one_hot=one_hot)
