# -*- coding: utf-8 -*-
# Author : Junhao Wei
# @file : test.py
# @Time : 2024/7/14 15:42
# @Function :

import json
import os.path

import joblib
import pandas as pd

y_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'Combined', 'no_one_hot', 'y_test.csv')
x_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'Combined', 'no_one_hot', 'x_test.csv')
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          'trained_model_info', 'CB', '2024-07-22_NoOnehot_f1-macro',
                          'retrain_CB_ROS_no_rif_MultiClassification',
                          '1_1_3',
                          'retrain_CB_ROS_no_rif_MultiClassification.pkl')
x, y = pd.read_csv(x_path), pd.read_csv(y_path)
x = x.drop(columns=['CASEWGT'])
model = joblib.load(model_path)
#
y_pred = model.predict(x)
y_proba = model.predict_proba(x)
for p in y_proba:
    print(
        sum(p)
    )
# class_weights = [{0: 1, 1: 1, 2: 2}, {0: 1, 1: 1, 2: 3}, {0: 1, 1: 1, 2: 4}, {0: 1, 1: 1, 2: 5}]
# for cls in class_weights:
#     print(cls)
