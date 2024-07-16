import numpy as np


##########
# Class-Conditional Conformal Prediction
# In CCCP, we calculate q_hat by class,
# then we get prediction sets with 1 - alpha confidence as:
# C(x_val) = {y: f(x_val)_y >= 1- q_hat^y}
##########

def calc_qhat_CCCP(y_softmax, y_cal, alpha=0.1):
    """
    CCCP q_hat approach for conformal prediction

    :param y_softmax: softmax output of a classifier.
    :param y_cal: labels of calibration data.
    :param alpha: User specified coverage requirement.

    :return: qhat, threshold score of the quantile.
    """
    y_cal = y_cal.tolist()
    calib_length = y_softmax.shape[0]
    q_hat_dict = {}
    class_softmax_dict = {}

    for i in range(calib_length):
        # Iterate calibration set and calc all uncertainty score by class.
        real_class = y_cal[i]
        uncertainty = 1 - y_softmax[i][real_class]

        if real_class not in class_softmax_dict:
            class_softmax_dict[real_class] = [uncertainty]
        else:
            class_softmax_dict[real_class].append(uncertainty)

    for clas, uncertainty_list in class_softmax_dict.items():
        # Calc q_hat by class.
        class_calib_length = len(uncertainty_list)
        q_hat_dict[clas] = np.quantile(uncertainty_list,
                                       np.ceil((class_calib_length + 1) * (1 - alpha)) / class_calib_length)
    return q_hat_dict


def get_confsets_CCCP(softmax_out, q_hat_dict):
    """
    APS conformal prediction. Key point is to find the q_hat by class.

    :param softmax_out: softmax out put of classification model;
    :param q_hat_dict: a dict of quantile q_hat for each class. (default as alpha=0.1).

    :return: [{prediction: probability}]. a prediction set with probability.
    """
    conf_set = []  # len(y) * {} for each x_val

    for i in range(softmax_out.shape[0]):
        pred_set = {}
        for cls in range(softmax_out.shape[1]):
            if 1 - softmax_out[i][cls] < q_hat_dict[cls]:
                pred_set[cls] = softmax_out[i][cls]
        conf_set.append(pred_set)

    return conf_set
