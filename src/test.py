# -*- coding: utf-8 -*-
# Author : Junhao Wei
# @file : test.py
# @Time : 2024/7/14 15:42
# @Function :

import json
import os.path

import joblib
import pandas as pd

y_path = ('F:\\Code_reposity\\PyProjects\\ConfPre2Post\\ConfPre2Post\\data\\Combined'
        '\\with_one_hot\\y_test.csv')
x_path = r'F:\Code_reposity\PyProjects\ConfPre2Post\ConfPre2Post\data\Combined\with_one_hot\x_test.csv'
model_path = (r'F:\Code_reposity\PyProjects\ConfPre2Post\ConfPre2Post\trained_mo'
              r'del_info\LGBM\2024-07-22_Onehot_f1-macro\retrain_LGBM_None_no_rif_Multi'
              r'Classification\weight_default\retrain_LGBM_None_no_rif_MultiClassification.pkl')
x, y = pd.read_csv(x_path), pd.read_csv(y_path)
model = joblib.load(model_path)

y_pred = model.predict(x)
print(y_pred)