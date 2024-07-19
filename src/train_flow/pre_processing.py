# -*- coding: utf-8 -*-
# Author : Junhao Wei
# @file : pre_processing.py
# @Time : 2024/7/14 17:30
# Interpretation: Flow of pre-processing.
import os.path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.pkts.my_logger import logger
from src.pkts.preprocessing_modules import CATEGORY_FEATURES, ALL_PRE_FEATURES


def get_processed_data(input_df: str, use_features: list, one_hot=False, cat_feats=CATEGORY_FEATURES):
    """
    Given input csv, do pre-processing,

    :param input_df: Input file path.
    :param use_features: Features for modeling.
    :param one_hot: Whether one-hot encoding for categorical features.
    :param cat_feats: Categorical feature list.

    :return: data with pre-processing (feature selection, fill null, resampling, etc.)
            as (features, labels, x_train, x_test, y_train, y_test)
                - features, labels: for cross-validation;
                - x_train, x_test, y_train, y_test: for re-train.
    """
    df_raw = pd.read_csv(input_df)
    df_raw = df_raw.replace(65536, np.nan)
    df_features = df_raw[use_features]
    df_features = df_features.reset_index(drop=True)

    for feat in df_features.columns:
        lack_count = df_features[feat].isnull().sum()
        if lack_count:
            logger.debug(f'缺失值:  {feat}, 其非缺失值比例： {round(1 - (lack_count / len(df_features)), 3)}')

    df_features = df_features.dropna(axis=0)
    severe_ratio = round(len(df_features[df_features['InjurySeverity'] == 2]) / len(df_features), 3)
    slight_ratio = round(len(df_features[df_features['InjurySeverity'] == 1]) / len(df_features), 3)
    logger.info(f'Drop null的数据重伤比： {severe_ratio}, 轻伤比：{slight_ratio}, 全样本数：{len(df_features)}')

    one_hot_str = 'no_one_hot'
    if one_hot:
        one_hot_str = 'with_one_hot'
        for feat in cat_feats:
            if feat in df_features.columns:
                df_features[feat] = df_features[feat].astype('category')
        data_onehot = pd.get_dummies(df_features[cat_feats], prefix=cat_feats)
        data_numeric = df_features.drop(columns=cat_feats)
        data_combined = pd.concat([data_onehot, data_numeric], axis=1)
        feats, labels = data_combined.drop(columns=['InjurySeverity']), data_combined['InjurySeverity']
    else:
        feats, labels = df_features.drop(columns=['InjurySeverity']), df_features['InjurySeverity']

    x_train, x_test, y_train, y_test = train_test_split(feats, labels, stratify=labels,
                                                        shuffle=True, test_size=0.15,
                                                        random_state=42)  # uniform for severe.

    assert round(len(y_train[y_train == 2]) / len(y_train), 3) == round(len(y_test[y_test == 2]) / len(y_test), 3), \
        'The proportion of severe samples in the training and testing sets is different！'

    # Save train test dataset
    train_test_data_dict = {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}
    train_test_save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                       'data', 'Combined', one_hot_str)
    os.makedirs(train_test_save_dir, exist_ok=True)
    for df_name, df in train_test_data_dict.items():
        df.to_csv(os.path.join(train_test_save_dir, f'{df_name}.csv'))

    return feats, labels, x_train, x_test, y_train, y_test


if __name__ == "__main__":
    input_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                             'data', 'Combined', 'CombineNassCiss.csv')
    get_processed_data(input_dir, use_features=ALL_PRE_FEATURES, one_hot=True)
