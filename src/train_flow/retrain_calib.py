"""
1. Prepare data;
2. Load best params;
3. Train model;
4. Save model & info;
5. Calibration;
6. Evaluation;
7. Combine CP result with test
"""
import json
import os

import pandas as pd

from src.pkts.retrain_modules import retrain
from src.train_flow.train_tab import retrain_tab_module
from src.eval_conformal_predictor import eval_with_cp
from src.pkts.preprocessing_modules import ALL_PRE_FEATURES


def total_flow(data_path, param_path, save_path, model_name, class_weight, use_feats):
    # 1. Prepare data
    df_raw = pd.read_csv(data_path, index_col=0)  # raw data, processing required
    for feat in use_feats:
        df_raw = df_raw[df_raw[feat] != 65536]


    # 2. Load best params
    with open(param_path, 'r', encoding='utf-8') as f:
        param_dict = json.load(fp=f)

    # 3~4. Train model & Save info
    if model_name in ['LGBM', 'CB', 'RF']:
        retrain(model_name, param_dict, x_train, y_train, x_test, y_test,
                sampling='ROS', class_weights=class_weight, save_dir=save_path)
    else:
        retrain_tab_module(x_train, x_test, y_train, y_test, param_dict,
                           sampling='ROS', save_dir=save_path)

    model_path = os.path.join()

    # 5~6. Calibration & Evaluation
    softmax_results = eval_with_cp(model_name, model_path, x_calib, y_calib, x_test, y_test,
                                   test_in_calib=False, cp_method='Naive', alpha=0.1)

    # 7. Combine CP result with test


if __name__ == '__main__':
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                             'data', 'Combined', 'CombineNassCiss.csv')
    use_feats = ALL_PRE_FEATURES
    use_feats.remove('CASEWGT')
