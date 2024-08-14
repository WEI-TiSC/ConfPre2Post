# -*- coding: utf-8 -*-
# Author : Junhao Wei
# @file : shap_anl.py
# @Time : 2024/8/14 13:59
# Interpretation of feature importance by SHAP
import os
import joblib
import pandas as pd
import shap
from matplotlib import pyplot as plt

if __name__ == "__main__":
    # Get data
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
                             'data', 'Combined', 'no_one_hot')
    x_test = pd.read_csv(os.path.join(data_path, 'x_test.csv'))
    if 'CASEWGT' in x_test.columns.values:
        x_test = x_test.drop(columns=['CASEWGT'])
    y_test = pd.read_csv(os.path.join(data_path, 'y_test.csv'))
    y_test = pd.Series(y_test['InjurySeverity'].values)

    # Read model
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
                              'trained_model_info', 'TabNet', '2024-07-23_ROS_f1_macro',
                              'Retrain_Tab_no_rif_ROS_MultiClassification', '1_1_3',
                              'Retrain_Tab_no_rif_ROS_MultiClassification.pkl')
    model = joblib.load(model_path)

    # Analysis path
    anl_path = os.path.join(os.path.dirname(model_path), 'analysis')
    os.makedirs(anl_path, exist_ok=True)

    # SHAP for interpretation
    background = shap.sample(x_test, 400)
    explainer = shap.KernelExplainer(model.predict_proba, background)  # Choose 400 cases for background
    shap_values = explainer.shap_values(x_test)

    inury_level = ['No Injury', 'Slight Injury', 'Severe Injury']

    for i in range(len(shap_values)):
        shap.summary_plot(shap_values[i], x_test,
                          feature_names=x_test.columns, max_display=28)  # Use all cases to analyze SHAP
        plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
        plt.title(f"SHAP summary plot of {inury_level[i]}")
        plt.savefig(os.path.join(anl_path, f"SHAP summary plot of {inury_level[i]}.png"),
                    dpi=300, bbox_inches='tight')
        plt.clf()
