import numpy as np
import pandas as pd


##########
# Naive CP flow: train -> calibration -> validation

# 1. Train model f: X -> (0, 1, 2) on Dtrain;
# 2. Define s(x, y) = 1 - f(x)_{ytrue} where f(x)_{ytrue} is the softmax output of true class;
# 3. Compute s1, s2, ..., sn_cal on the calibration set Dcal;
# 4. Compute qhat as the (n_cal + 1)(1-a)/n_cal quantile of the scores;
# 5. Predict with 1-a confidence as C(x_val) = {y: f(x_val)_{y} >= 1 - hat_q}
##########

def calc_qhat_naive_classification(y_softmax, y_cal: pd.Series, alpha=0.1):
    """
    Naive approach for conformal prediction

    :param y_softmax: softmax output of a classifier.
    :param y_cal: labels of calibration data.
    :param alpha: User specified coverage requirement.

    :return: qhat, threshold score of the quantile.
    """
    y_cal = y_cal.tolist()
    calib_length = y_softmax.shape[0]
    scores = np.zeros(calib_length)

    for i in range(calib_length):  # calc all scores
        softmax_true_class = y_softmax[i][y_cal[i]]
        scores[i] = 1 - softmax_true_class

    q_hat = np.quantile(scores, np.ceil((calib_length + 1) * (1 - alpha))/calib_length)
    return q_hat


def get_confsets_naive(softmax_out, q_hat):
    """
    Naive conformal prediction.

    :param softmax_out: softmax out put of classification model;
    :param q_hat: quantile q (default as alpha=0.1).

    :return: [{prediction: probability}]. a prediction set with probability.
    """
    conf_set = []  # len(y) * {} for each x_val

    for i in range(softmax_out.shape[0]):
        assert round(sum(softmax_out[i]), 3) == 1.00, f"Error: sum of softmax output not 1.00! {softmax_out[i]}"
        pred_set = {}
        for j in range(softmax_out.shape[1]):  # Check each output in cur sample
            if softmax_out[i][j] >= 1 - q_hat:
                pred_set[j] = softmax_out[i][j]
        conf_set.append(pred_set)

    return conf_set
