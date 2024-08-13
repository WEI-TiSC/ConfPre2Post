import numpy as np

from src.pkts.my_logger import logger


##########
# Adaptive Prediction Set (APS)

# Key:
# 1. Choose s(x,y) = SUM_{i=1}^k pi(x)_y till we reach the true class. Which is, the correct answer always in set!
# 2. s(x,y)= SUM_{i=1}^k hat_pi(x)_y, and compute the quantile hat_q.
# 3. Make prediction with confidence 1-alpha as:
#  C(x_val) = {y: SUM_{i=1}^k pi(x)_y >= hat_q}
##########
def get_true_idx(y_pred, y_true):
    """
    Find the minimum INDEX which contains the true label

        e.g. y_pred = [0.12, 0.48, 0.19, 0.1, 0.11]
             y_true = 2
             return is: 1 (for get idx 0 and 1)

    :param y_pred: ONE softmax output.
    :param y_true: ONE true label.
    :return: true index (int)
    """
    return np.where(np.sort(y_pred)[::-1] == y_pred[y_true])[0][0]


def calc_qhat_aps(y_softmax, y_cal, alpha=0.1):
    """
    Find quantile q_hat in APS method.
    """
    y_cal = y_cal.tolist()
    sum_densitys = []
    calib_length = y_softmax.shape[0]

    for i in range(calib_length):
        idx = get_true_idx(y_pred=y_softmax[i], y_true=y_cal[i])
        sum_densitys.append(np.sum(np.sort(y_softmax[i])[::-1][0: idx + 1]))  # Calc sum of softmax as the s(x, y)

    worst_prediction = 0
    for i in range(len(sum_densitys)):
        if round(sum_densitys[i], 3) >= 1.000:
            worst_prediction += 1
    logger.warning("%s cases out of total %s cases are worst prediction!" % (worst_prediction, len(sum_densitys)))  # 真实结果在最后就是废的
    q_hat = np.quantile(sum_densitys, np.ceil((calib_length + 1) * (1 - alpha)) / calib_length)
    return q_hat


def get_confsets_APS(softmax_out, q_hat):
    """
    APS conformal prediction. Key point is to find the minumum length while keeping cumsum > q_hat.

    :param softmax_out: softmax out put of classification model;
    :param q_hat: quantile q (default as alpha=0.1).

    :return: [{prediction: probability}]. a prediction set with probability.
    """
    conf_set = []  # len(y) * {} for each x_val

    for i in range(softmax_out.shape[0]):  # 找到累加值大于q_hat的最小set size
        cur_softmax = softmax_out[i]
        idx = np.where(np.cumsum(np.sort(cur_softmax)[::-1]) >= q_hat - 0.001)  # 减去以避免q_hat=1.0
        idx = idx[0][0]  # cumsum: 累加（即APS的softmax值累加）
        pred_set = {}

        for j in range(idx + 1):
            app_class_idx = np.where(cur_softmax == np.sort(cur_softmax)[::-1][j])[0][0]
            class_prob = cur_softmax[app_class_idx]
            pred_set[app_class_idx] = class_prob

        conf_set.append(pred_set)
    return conf_set


if __name__ == "__main__":
    y_1 = [[0.12, 0.48, 0.19, 0.1, 0.11], [0.1, 0.1, 0.1, 0.3, 0.4]]
    y_true = [2, 4]
    print(np.where(np.cumsum(np.sort(y_1[0])[::-1]) > 0.7))
