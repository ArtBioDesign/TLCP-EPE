# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Config(object):

    """配置参数"""
    def __init__(self):
        self.model_name = 'MLP'
        self.class_list = list(range(2))             
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
        self.dropout = 0.1                                          
        self.num_classes = 2                        
        self.embed = 1024 
        self.hidden_size = 1792                                     
        self.num_layers = 2                                          
        self.hidden_size2 = 64


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        self.fc1 = nn.Linear(config.embed, config.hidden_size)
        self.bn1 = nn.BatchNorm1d(config.hidden_size)  # 添加BN
        self.dropout = nn.Dropout(config.dropout) 
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        x = x.squeeze(1)
        out = self.fc1(x) 
        out = self.bn1(out)  # BN
        out = F.relu(out)
        out = self.dropout(out)  # Dropout
        out = self.fc(out)  # [128, 64]
        return out

