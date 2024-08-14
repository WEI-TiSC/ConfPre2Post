# -*- coding: utf-8 -*-
# Author : Junhao Wei
# @file : tab_attention_anl.py
# @Time : 2024/8/14 12:36
# Interpretation of Attention Module
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':
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

    # Get normalized attention weights
    explainer = model.explain(x_test.values)
    masks = explainer[0]  # Attention values
    attention_weights = masks.mean(axis=0).reshape(1, -1)  # Average
    min_val = attention_weights.min()
    max_val = attention_weights.max()
    normalized_weights = (attention_weights - min_val) / (max_val - min_val)
    normalized_weights = normalized_weights.T  # shape: (28, 1)

    # Visualization
    plt.figure(figsize=(6, 6))
    plt.imshow(normalized_weights, cmap='plasma', interpolation='nearest', aspect=0.3)
    plt.colorbar(shrink=0.8)
    plt.title("Normalized TabNet Attention Weights")
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.yticks(ticks=np.arange(28), labels=[f'{x_test.columns[i]}' for i in range(28)])
    plt.xticks([])
    plt.savefig(os.path.join(anl_path, 'normalized_tabnet_attention_weight.png'),
                              dpi=300, bbox_inches='tight')
    plt.show()

    # Descending sort attention weights
    sorted_indices = np.argsort(normalized_weights[:, 0])[::-1]
    sorted_feat_names = np.array(x_test.columns)[sorted_indices]
    sorted_weights = normalized_weights[sorted_indices]

    sorted_data = np.column_stack((sorted_feat_names, sorted_weights))
    np.savetxt(os.path.join(anl_path, 'sorted_normalized_weights.csv'), sorted_data, fmt='%s,%.3f',
               delimiter=',', header='Feature, Importance', comments='')
    print("Sorted Features and Weights")
    for name, weight in sorted_data:
        print(f"{name}: {float(weight):.3f}")
