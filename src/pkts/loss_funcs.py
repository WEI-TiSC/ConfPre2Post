import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2, num_classes=3, size_average=True):
        """
        FocalLoss = -a(1-yi)**gamma * ce_loss(xi, yi)

        :param alpha: 类别权重，实际使用compute_class_weight获得;
        :param gamma: 难以样本调节程度，默认为2;
        :param num_classes: 类别总数，此处问题为3;
        :param size_average: 是否取损失平均值，默认为True.
        """
        super(FocalLoss, self).__init__()
        self.size_average = size_average
        self.gamma = gamma
        if alpha == None:
            self.alpha = torch.ones(num_classes)  # alpha为空则默认平等权重
        else:
            if isinstance(alpha, list):
                assert len(alpha) == num_classes  # alpha也可以是list，size: [num_classes], 对不同类别赋权
                self.alpha = torch.Tensor(alpha)
            else:
                self.alpha = alpha  # 输入alpha是Tensor则直接赋权

    def forward(self, preds, labels):
        """
        :param preds: 预测值。     size: [B, C]  B:batch, C: num_classes
        :param labels: 真实标签。  size: [B]
        """
        preds = preds.view(-1, preds.size(-1))
        alpha = self.alpha.to(preds.device)

        ##########
        # focalLoss(pt) = -alpha * (1-py)^y * log(pt)
        ##########
        pt = F.softmax(preds, dim=1)
        log_pt = torch.log(pt)

        ##########
        # 选择真实标签对应的数据！
        ##########
        labelsView = labels.view(-1, 1).long()
        pt = pt.gather(1, labelsView)
        log_pt = log_pt.gather(1, labelsView)

        ##########
        # 组装
        ##########
        loss_without_alpha = -torch.mul(torch.pow((1-pt), self.gamma), log_pt)
        label_flatten = labelsView.view(-1)
        alpha = alpha.gather(0, label_flatten)
        loss = torch.mul(alpha, loss_without_alpha.t())

        # 如果 self.size_average=True 则算平均否之算和
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()

        return loss
