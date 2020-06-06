import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone.resnet import res50, res101,res152
from backbone.resnet_cifar import res32_cifar
from backbone.classifier import *
from backbone.resnest.torch.resnest import resnest50, resnest101, resnest200, resnest269


class Network(nn.Module):
    def __init__(self, cfg):
        super(Network, self).__init__()

        self.num_classes = cfg.num_classes
        self.cfg = cfg
        if 'resnest' in self.cfg.BACKBONE:
            self.backbone = eval(self.cfg.BACKBONE)()
        else:
            self.backbone = eval(self.cfg.BACKBONE)(
                self.cfg,
                pretrained_model=cfg.PRETRAINED_BACKBONE,
                last_layer_stride=2,
            )
        self.feature_len = self.get_feature_length()
        # print(self.feature_len, self.num_classes)
        if self.cfg.CLASSIFIER == 'FC':
            self.classifier = nn.Linear(self.feature_len,
                                        self.num_classes,
                                        bias=True)
        else:
            self.classifier = eval(self.cfg.CLASSIFIER)(
                num_features=self.feature_len, num_classes=self.num_classes)

    def freeze_backbone(self):
        print("Freezing backbone .......")
        for p in self.backbone.parameters():
            p.requires_grad = False

    def load_backbone_model(self, backbone_path=""):
        self.backbone.load_model(backbone_path)
        print("Backbone has been loaded...")

    def load_model(self, model_path):
        pretrain_dict = torch.load(model_path, map_location="cuda")
        pretrain_dict = pretrain_dict[
            'state_dict'] if 'state_dict' in pretrain_dict else pretrain_dict
        model_dict = self.state_dict()
        from collections import OrderedDict
        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if k.startswith("module"):
                new_dict[k[7:]] = v
            else:
                new_dict[k] = v
        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        print("Model has been loaded...")

    def get_feature_length(self):
        if "cifar" in self.cfg.BACKBONE:
            num_features = 64
        else:
            num_features = 2048
        return num_features

    def forward(self, x, **kwargs):
        x = self.backbone(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

# model = resnest50()
# print(model)