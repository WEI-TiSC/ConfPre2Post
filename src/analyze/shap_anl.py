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
    background = shap.sample(x_test, 30)
    explainer = shap.KernelExplainer(model.predict_proba, background)  # Choose 400 cases for background
    shap_values = explainer.shap_values(x_test[:15])

    inury_level = ['No Injury', 'Slight Injury', 'Severe Injury']

    # Save beeswarm
    beeswarm_path = os.path.join(anl_path, 'beeswarm')
    os.makedirs(beeswarm_path, exist_ok=True)
    for i in range(len(shap_values)):
        shap.summary_plot(shap_values[i], x_test[:15],
                          feature_names=x_test.columns, max_display=28, show=False
                          )
        plt.savefig(os.path.join(beeswarm_path, f"SHAP_summary_plot_of_{inury_level[i]}.png"),
                    dpi=300, bbox_inches='tight')
        plt.show()

    # Save heatmap
    heatmap_path = os.path.join(anl_path, 'heatmap')
    os.makedirs(heatmap_path, exist_ok=True)

    # Only draw severe SHAP heatmap?
    for i in range(len(shap_values)):
        draw_injury_level = i
        shap_values_expl = shap.Explanation(
            values=shap_values[draw_injury_level],
            base_values=explainer.expected_value[draw_injury_level],
            data=x_test[:15],
            feature_names=x_test.columns
        )
        shap.plots.heatmap(shap_values_expl, max_display=28, show=False, plot_width=13.0)
        plt.savefig(os.path.join(heatmap_path, f"SHAP_heatmap_for_{inury_level[i]}.png"), dpi=300, bbox_inches='tight')
        plt.gca().tick_params(axis='y', labelsize=7)
        plt.show()

    # TODO: change shap cases!

"""
f(x) 曲线：

图上方的 f(x) 曲线表示模型的输出值（预测值）的变化情况。对于每个样本（或实例），f(x) 代表模型的预测值。当你处理的是一个回归任务时，f(x) 可能是一个连续值；对于分类任务，它可能是某个类别的预测概率。
右方的柱状图：

右侧的黑色柱状图表示每个特征在所有样本中的总体重要性。这些柱状图显示了每个特征在多个样本中 SHAP 值的总和或平均值。柱状图越长，表示该特征对模型输出的影响越大。
"""
