# -*- coding: utf-8 -*-
# Author : Junhao Wei
# @file : test.py
# @Time : 2024/7/14 15:42
# @Function :

import json
import os.path

import joblib
import numpy as np
import pandas as pd
import torch

from src.pkts.preprocessing_modules import ALL_PRE_FEATURES, CATEGORY_FEATURES

df_path = r'F:\Code_reposity\PyProjects\ConfPre2Post\ConfPre2Post\data\Combined\CombineNassCiss.csv'


df = pd.read_csv(df_path)
df = df[ALL_PRE_FEATURES]
df = df.replace(65536, np.nan)
df = df.dropna(axis=0)
df = df.reset_index(drop=True)

inj_seve = df['InjurySeverity'].value_counts()
for k, v in inj_seve.items():
    print(f"inj seve {k} with {v} cases, perc {round(v/len(df), 4)}")

# 看特征百分比
for feat in df.columns.values:
    if feat not in CATEGORY_FEATURES: continue
    tmp_percentage = df[feat].value_counts()
    for k, v in tmp_percentage.items():
        satisfied = df.loc[df[feat] == k]
        counts_severe = satisfied.loc[satisfied['InjurySeverity']==2]
        perc_severe = len(counts_severe) / len(satisfied)
        counts_slight = satisfied.loc[satisfied['InjurySeverity']==1]
        perc_slight = len(counts_slight) / len(satisfied)
        print(f'Feature:{feat} with value {k} has {v} cases, slight perc: {round(perc_slight, 4)}, severe perc: {round(perc_severe, 4)}')

