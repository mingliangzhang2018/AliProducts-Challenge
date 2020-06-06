import time
import os
import torch
# from utils.lr_scheduler import WarmupMultiStepLR
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt


class CrossEntropy(nn.Module):
    def __init__(self, para_dict=None):
        super(CrossEntropy, self).__init__()

    def forward(self, output, target):
        output = output
        loss = F.cross_entropy(output, target)
        return loss


def get_optimizer(cfg, model):

    params = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            params.append({"params": p})
            #print(name)
    if cfg.OPTIMIZER == "SGD":
        optimizer = torch.optim.SGD(
            # [{'params': model.backbone.parameters(), 'lr': 1e-5},\
            # {'params': model.classifier.parameters(), 'lr': cfg.BASE_LR}],\
            params,
            lr=cfg.BASE_LR,
            momentum=0.9,
            weight_decay=cfg.WEIGHT_DECAY,)
            #nesterov=True)
    elif cfg.OPTIMIZER == "ADAM":
        optimizer = torch.optim.Adam(
            params,
            lr=cfg.BASE_LR,
            betas=(0.9, 0.999),
            weight_decay=cfg.WEIGHT_DECAY,
        )
    else:
        raise NotImplementedError("Unsupported optimizer: {}".format(
            cfg.OPTIMIZER))

    return optimizer


def get_scheduler(cfg, optimizer):
    if cfg.LR_SCHEDULER == "multistep":
        # step = int(cfg.MAX_EPOCH / 4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            cfg.DECAY_STEP,
            # [10, step, step * 2, step * 3],
            gamma=0.1,
        )
    elif cfg.LR_SCHEDULER == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.MAX_EPOCH, eta_min=1e-5)
    else:
        raise NotImplementedError("Unsupported LR Scheduler: {}".format(
            cfg.LR_SCHEDULER))

    return scheduler


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


class FusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)

    def update(self, output, label):
        length = output.shape[0]
        for i in range(length):
            self.matrix[output[i], label[i]] += 1

    def get_rec_per_class(self):
        rec = np.array([
            self.matrix[i, i] / self.matrix[:, i].sum()
            for i in range(self.num_classes)
        ])
        rec[np.isnan(rec)] = 0
        return rec

    def get_pre_per_class(self):
        pre = np.array([
            self.matrix[i, i] / self.matrix[i, :].sum()
            for i in range(self.num_classes)
        ])
        pre[np.isnan(pre)] = 0
        return pre

    def get_accuracy(self):
        acc = (np.sum([self.matrix[i, i]
                       for i in range(self.num_classes)]) / self.matrix.sum())
        return acc

    def plot_confusion_matrix(self, normalize=False, cmap=plt.cm.Blues):

        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = self.matrix.T

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            # ... and label them with the respective list entries
            xticklabels=np.arange(self.num_classes),
            yticklabels=np.arange(self.num_classes),
            title=title,
            ylabel='True label',
            xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(),
                 rotation=45,
                 ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j,
                        i,
                        format(cm[i, j], fmt),
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return fig


def accuracy(output, label):
    cnt = label.shape[0]
    true_count = (output == label).sum()
    now_accuracy = true_count / cnt
    return now_accuracy, cnt


# class CSCE(nn.Module):

#      def __init__(self, para_dict=None):
#           super(CSCE, self).__init__()
#           self.num_class_list = para_dict["num_class_list"]
#           self.device = para_dict["device"]

#           cfg = para_dict["cfg"]
#           scheduler = cfg.LOSS.CSCE.SCHEDULER
#           self.step_epoch = cfg.LOSS.CSCE.DRW_EPOCH

#           if scheduler == "drw":
#                self.betas = [0, 0.999999]
#           elif scheduler == "default":
#                self.betas = [0.999999, 0.999999]
#           self.weight = None

#      def update_weight(self, beta):
#           effective_num = 1.0 - np.power(beta, self.num_class_list)
#           per_cls_weights = (1.0 - beta) / np.array(effective_num)
#           per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.num_class_list)
#           self.weight = torch.FloatTensor(per_cls_weights).to(self.device)

#      def reset_epoch(self, epoch):
#           idx = (epoch-1) // self.step_epoch
#           beta = self.betas[idx]
#           self.update_weight(beta)

#      def forward(self, x, target, **kwargs):
#           return F.cross_entropy(x, target, weight= self.weight)

# # The LDAMLoss class is copied from the official PyTorch implementation in LDAM (https://github.com/kaidic/LDAM-DRW).
# class LDAMLoss(nn.Module):

#      def __init__(self, para_dict=None):
#           super(LDAMLoss, self).__init__()
#           s = 30
#           self.num_class_list = para_dict["num_class_list"]
#           self.device = para_dict["device"]

#           cfg = para_dict["cfg"]
#           max_m = cfg.LOSS.LDAM.MAX_MARGIN
#           m_list = 1.0 / np.sqrt(np.sqrt(self.num_class_list))
#           m_list = m_list * (max_m / np.max(m_list))
#           m_list = torch.FloatTensor(m_list).to(self.device)
#           self.m_list = m_list
#           assert s > 0

#           self.s = s
#           self.step_epoch = cfg.LOSS.LDAM.DRW_EPOCH
#           self.weight = None

#     def reset_epoch(self, epoch):
#           idx = (epoch-1) // self.step_epoch
#           betas = [0, 0.9999]
#           effective_num = 1.0 - np.power(betas[idx], self.num_class_list)
#           per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
#           per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.num_class_list)
#           self.weight = torch.FloatTensor(per_cls_weights).to(self.device)

#     def forward(self, x, target):
#           index = torch.zeros_like(x, dtype=torch.uint8)
#           index.scatter_(1, target.data.view(-1, 1), 1)

#           index_float = index.type(torch.FloatTensor)
#           index_float = index_float.to(self.device)
#           batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
#           batch_m = batch_m.view((-1, 1))
#           x_m = x - batch_m

#           output = torch.where(index, x_m, x)
#           return F.cross_entropy(self.s * output, target, weight= self.weight)