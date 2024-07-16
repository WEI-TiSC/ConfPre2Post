# -*- coding: utf-8 -*-
# Author : Junhao Wei
# @file : preprocessing_modules.py
# @Time : 2024/7/14 15:42
# Interpretation: This file is for data pre-processing, including data split, resampling(if does),
# feature selection, etc.
import numpy as np
import pandas as pd
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek

from src.pkts.my_logger import logger

SAMPLING_POOL = {
    # Over Sampling
    'ADASYN': ADASYN(random_state=42, n_neighbors=10, sampling_strategy='not majority'),
    'BSMOTE': BorderlineSMOTE(random_state=42, sampling_strategy=1),
    'SMOTETomek': SMOTETomek(random_state=42, sampling_strategy='not majority'),

    # Under Sampling
    'TomekLinks': TomekLinks(sampling_strategy='majority')
}

FACTOR_DICT = {
    "analyze": [
        'Maneuver before collision',
        'Crash Type',  # Disccusion by crash type
        'delta v confidence level',
        'delta v',  # null: 2862
        'MAIS3+',
        # Pre-existing conditions which might affect risk injury.
        'CARDIOCOND',  # pre-existing cardiovascular condition;
        'SPINEDEGEN',  # pre-existing degenerative spinal condition;
        'IMPAIREDCOAG',  # pre-existing impaired coagulation condition;
        'IMPLANTFUS',  # history of musculoskeletal implant, surgery, or fusion;
        'OSTEOCOND',  # a pre-existing history of osteoporosis or osteopenia;
        'COMORBOTH',  # pre-existing comorbidity that isn’t captured in the other comorbidity variables.
    ],

    "features": {
        'Driver':
            [
                'Alcohol Present',  # null: 1031
                'Distracted in Driving',  # null: 2481
                'height',
                'weight',
                'Age',
                'Sex',
                'Race',
            ],

        'Vehicle':
            [
                'premovement before collision',
                'Body Category',  # 与Curb Weight有点重合感
                'Pre-event Location',
                'Curb Weight',  # 与Body Category有点重合感
                'Model Year',
                'Clock-form Direction of force'
            ],

        'TrafEnv':
            [
                'Speed Limit',
                'Number of lanes',
                'Traffic Condition',
                'Surface Condition',
                'Surface Type',
                'Uphill or Downhill',
                'Crash Type',
                'Lighting Condition',
                'Related to Intersection',
                'Traffic Conrtol Functioning',
                'Climate',
                'Alignment of Road'
            ],
        'rif':
            ['CASEWGT'],

        "Time_serie": ['year', 'month', 'Day in Week'],

        'Object': ['InjurySeverity'],
        # 'OtherVeh':
        #     [  # 此类内都可以不使用！（因为缺失多，且不一定能事前获得）
        #         'Other Veh Body Category',
        #         'Other Veh premovement before collision',
        #         'Other Veh Clock-form Direction of force'
        #     ]
    },
}


ALL_PRE_FEATURES = []
for _, val in FACTOR_DICT.get('features').items():
    for feat in val:
        ALL_PRE_FEATURES.append(feat)


CATEGORY_FEATURES = [
    'Race', 'Sex', 'Alcohol Present', 'Distracted in Driving', 'premovement before collision',
    'Body Category', 'Pre-event Location', 'Surface Condition', 'Traffic Condition',
    'Surface Type', 'Uphill or Downhill', 'Crash Type', 'Lighting Condition', 'Related to Intersection',
    'Traffic Conrtol Functioning', 'Climate', 'Alignment of Road',
    ]


def rif_resampling(df: pd.DataFrame, rif_col='CASEWGT') -> pd.DataFrame:
    """
    Define rif resampling

    :param df: data input
    :param rif_col: rif based column
    :return: rif_ed data as (rif_x, rif_y).
    """
    rif_data = {}
    data = df.reset_index(drop=True)
    for num, col in enumerate(data.columns.values):
        if col == rif_col:
            continue
        rif_data[col] = []
        for i in data.index:
            rif_factor = round(data.loc[i, rif_col])
            val = data.loc[i, col]
            rif_data[col].extend([val for _ in range(rif_factor)])
    logger.info('Processing Finished...Now starting combining dataframe')
    rif_df = pd.DataFrame({col_name: feat_val for col_name, feat_val in rif_data.items()})
    logger.info('Dataframe created! Now returning data...')
    return rif_df


def run_rif(features: pd.DataFrame, labels: pd.Series) -> (pd.DataFrame, pd.Series):
    combined_df = pd.concat([features, labels], axis=1)
    rifed_data = rif_resampling(combined_df)

    # NOTE: RIF takes a lot of time!
    x_rif, y_rif = rifed_data.drop(columns=['InjurySeverity']), rifed_data['InjurySeverity']
    return x_rif, y_rif


def data_resampling(features: pd.DataFrame, labels: pd.Series, sampling_method: str) -> (pd.DataFrame, pd.Series):
    """

    :param labels: labels
    :param features: features
    :param sampling_method: as the name.
    :return: resampled X/y
    """
    assert sampling_method in SAMPLING_POOL.keys(), 'Unknown resampling technique!'
    sampler = SAMPLING_POOL.get(sampling_method)
    x_res, y_res = sampler.fit_resample(features, labels)
    return x_res, y_res
