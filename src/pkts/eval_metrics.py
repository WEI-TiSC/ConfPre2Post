import datetime
import os
import numpy as np

from pytorch_tabnet.metrics import Metric
import matplotlib.pyplot as plt
import sklearn.metrics as metrics


def draw_confusion_matrix(y_test, y_pred, save_path, model_info):
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                                          display_labels=['No Injury', 'Slight Injury', 'Severe Injury'])
    disp.plot()

    plt.savefig(os.path.join(save_path, f'confusion matrix of {model_info}.png'))
    # plt.show()


def draw_roc_curve(y_test, y_proba, save_path,model_info):
    """
    仅对二分类问题适用
    """
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_proba[:, 1], pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)

    plt.plot(fpr, tpr, 'b', label='AUC = %0.3f' % roc_auc)
    plt.legend(loc='lower right')
    plt.xlim([0, 1.1])
    plt.ylim([0, 1.1])
    plt.xlabel = "False Positive Rate"
    plt.ylabel = "True Positive Rate"
    plt.title(f"Receiver Operating Characteristic of {model_info}")

    plt.savefig(os.path.join(save_path, f'ROC of {model_info}.png'))
    # plt.show()


def multi_models_roc(models, preds, y_label, colors=['crimson', 'blue', 'green'], save=True, dpin=100):
    """
    将多个机器模型的roc图输出到一张图上（仅对二分类适用）

    Args:
        models: list, 多个模型的名称
        preds: list, 多个模型的实例化对象
        save: 选择是否将结果保存（默认为png格式）

    Returns:
        返回图片对象plt
    """
    plt.figure(figsize=(10, 10), dpi=dpin)

    for (model_sampling, y_pred, colorname) in zip(models, preds, colors):
        fpr, tpr, thresholds = metrics.roc_curve(y_label, y_pred, pos_label=1)

        plt.plot(fpr, tpr, lw=5, label='{} (AUC={:.3f})'.format(model_sampling,
                                                                metrics.auc(fpr, tpr)), color=colorname)
        plt.plot([0, 1], [0, 1], '--', lw=5, color='grey')
        plt.axis('square')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate', fontsize=20)
        plt.ylabel('True Positive Rate', fontsize=20)
        plt.title('ROC Curve', fontsize=25)
        plt.legend(loc='lower right', fontsize=20)

    if save:
        plt.savefig(os.path.join(os.getcwd(), f'figs/{datetime.date.today()}_multi_models_roc.png'))
    return plt


class f1_macro_for_eval(Metric):
    def __init__(self, ):
        self._name = 'f1_macro_for eval'
        self._maximize = True

    def __call__(self, y_true, y_pred):
        y_pred = np.argmax(y_pred, axis=1)
        return metrics.f1_score(y_true=y_true, y_pred=y_pred, average='macro')