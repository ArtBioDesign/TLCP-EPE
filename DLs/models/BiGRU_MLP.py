# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Config(object):
    """配置参数"""
    def __init__(self):
        self.model_name = 'BiGRU_MLP'
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

        self.lstm = nn.GRU(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        
        # 直接使用GRU输出的处理
        self.fc1 = nn.Linear(config.hidden_size * 2, config.hidden_size2)
        self.bn1 = nn.BatchNorm1d(config.hidden_size2)
        self.dropout = nn.Dropout(config.dropout) 
        self.fc = nn.Linear(config.hidden_size2, config.num_classes)

    def forward(self, x):
        # GRU输出
        H, _ = self.lstm(x)  # [batch_size, seq_len, hidden_size * 2]

        # 去掉Attention机制，直接聚合序列信息
        # 方法1：平均池化（推荐）
        out = H.mean(dim=1)  # [batch_size, hidden_size * 2]
        
        # 分类
        out = F.relu(out)
        # out = self.fc1(out) 
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc(out)
        
        return out