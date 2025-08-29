# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Config(object):

    """配置参数"""
    def __init__(self):
        self.model_name = 'BiGRU_Attention'
        self.class_list = list(range(2))             
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
        self.dropout = 0.1                                          
        self.num_classes = 2                        
        self.embed = 1792 
        self.hidden_size = 1792                                     
        self.num_layers = 2                                           
        self.hidden_size2 = 64


'''Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification'''

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        self.lstm = nn.GRU(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.tanh1 = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(config.hidden_size * 2))
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(config.hidden_size * 2, config.hidden_size2)
        self.bn1 = nn.BatchNorm1d(config.hidden_size2)  # 添加BN
        self.dropout = nn.Dropout(config.dropout) 
        self.fc = nn.Linear(config.hidden_size2, config.num_classes)

    def forward(self, x):
        H, _ = self.lstm(x)  # [batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256] 
        M = self.tanh1(H)  # [128, 32, 256]
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [128, 32, 1]
        out = H * alpha  # [128, 32, 256]
        out = torch.sum(out, 1)  # [128, 256]
        out = F.relu(out)
        out = self.fc1(out) 
        out = self.bn1(out)  # BN
        out = F.relu(out)
        out = self.dropout(out)  # Dropout
        out = self.fc(out)  # [128, 64]
        return out
    