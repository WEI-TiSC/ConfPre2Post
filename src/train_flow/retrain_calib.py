"""
NOTE: Not used for models are all well-trained and only calibration is required.


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
from sklearn.model_selection import train_test_split

from src.pkts.retrain_modules import retrain
from src.train_flow.train_tab import retrain_tab_module
from src.eval_conformal_predictor import eval_with_cp
from src.pkts.preprocessing_modules import ALL_PRE_FEATURES


def total_flow(data_path, param_path, save_path, model_name, class_weight, use_feats):
    """
    Train -> Classification_Test -> CP_Calibration -> CP_Evaluation
    :param data_path: Path for data
    :param param_path: Path for hyper-params
    :param save_path: Path for saving results
    :param model_name: Model used
    :param class_weight: Given class weight
    :param use_feats: Features used
    :return: None
    """
    # 1. Prepare data
    df_raw = pd.read_csv(data_path, index_col=0)  # raw data, processing required
    for feat in use_feats:
        df_raw = df_raw[df_raw[feat] != 65536]  # del null for all features

    feats, labels = df_raw.drop(columns=['InjurySeverity']), df_raw['InjurySeverity']
    x_train, x_test_anl, y_train, y_test_anl = train_test_split(feats, labels, stratify=labels,
                                                                            shuffle=True, test_size=0.2,
                                                                            random_state=42)
    x_train = x_train[use_feats]  # 80%: 3:1 for train: calib

    x_test = x_test_anl[use_feats]  # Mapping! Concat other results with x_test_anl later.
    y_test = y_test_anl

    # 2. Load best params
    with open(param_path, 'r', encoding='utf-8') as f:
        param_dict = json.load(fp=f)

    # 3~4. Train model & Save info
    if model_name in ['LGBM', 'CB', 'RF']:
        model_path, model_info = retrain(model_name, param_dict, x_train, y_train, x_test, y_test,
                                         sampling='ROS', class_weights=class_weight, save_dir=save_path)
    else:
        model_path, model_info = retrain_tab_module(x_train, x_test, y_train, y_test, param_dict,
                                                    sampling='ROS', save_dir=save_path)

    full_model_path = os.path.join(model_path, f'{model_info}.pkl')

    # 5~6. Calibration & Evaluation
    softmax_series = eval_with_cp(model_name, full_model_path, x_train, y_train, x_test, y_test,
                                  test_in_calib=True, cp_method='Naive', alpha=0.1)
    set_size = softmax_series.apply(len)  # Add set size!
    # covered = pd.Series([1 if y_test_anl[i] in softmax_series[i].keys() else 0 for i in enumerate(softmax_series)])

    # 7. Combine CP result with test
    analysis_data = pd.concat([x_test_anl, y_test_anl, softmax_series, set_size], axis=1)
    analysis_data.to_csv(os.path.join(save_path, 'analysis_data.csv'), index=False)
    # TODO: find covered cases in Jupyter!


if __name__ == '__main__':
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                             'data', 'Combined', 'CombineNassCiss.csv')
    use_feats = ALL_PRE_FEATURES
    use_feats.remove('CASEWGT')
    use_feats.remove('InjurySeverity')

    model_name = 'TabNet'
    pre_train_info = '2024-07-23_ROS_f1_macro'
    param_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                             'trained_model_info', model_name, pre_train_info, 'pre_train_info', 'param_dict.json')
    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                             'ret_calib', model_name)
    os.makedirs(save_path, exist_ok=True)

    class_weight = {0: 1, 1: 1, 2: 3}  # For RF Onehot ROS -- Onehot Not Done!!!
    total_flow(data_path, param_path, save_path, model_name, class_weight, use_feats)

