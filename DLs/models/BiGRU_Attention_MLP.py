# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn import MultiheadAttention


class Config(object):

    """配置参数"""
    def __init__(self):
        self.model_name = 'BiGRU_Attention_MLP'
        self.class_list = list(range(2))             
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
        self.dropout = 0.1                                          
        self.num_classes = 2                        
        self.embed = 1024 
        self.hidden_size = 1792                                     
        self.num_layers = 2                                          
        self.hidden_size2 = 64

class ScaledDotProductAttention(nn.Module):
    def __init__(self, hidden_size):
        super(ScaledDotProductAttention, self).__init__()
        self.hidden_size = hidden_size
        self.scale = np.sqrt(hidden_size)
        
        # 可学习的查询向量
        self.query_vector = nn.Parameter(torch.randn(hidden_size))
        
    def forward(self, H):
        # H: [batch_size, seq_len, hidden_size]
        # 使用固定的查询向量与所有键计算相似度
        query = self.query_vector.unsqueeze(0).unsqueeze(0)  # [1, 1, H]
        query = query.expand(H.size(0), 1, -1)  # [B, 1, H]
        
        # 计算点积分数
        scores = torch.matmul(query, H.transpose(1, 2)) / self.scale  # [B, 1, L]
        scores = scores.squeeze(1)  # [B, L]
        
        weights = F.softmax(scores, dim=1)  # [B, L]
        weights = weights.unsqueeze(-1)  # [B, L, 1]
        context = torch.sum(H * weights, dim=1)  # [B, H]
        return context

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.lstm = nn.GRU(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        
        self.attention = ScaledDotProductAttention(config.hidden_size * 2)
        
        self.fc1 = nn.Linear(config.hidden_size * 2, config.hidden_size2)
        self.bn1 = nn.BatchNorm1d(config.hidden_size2)
        self.dropout = nn.Dropout(config.dropout) 
        self.fc = nn.Linear(config.hidden_size2, config.num_classes)

    def forward(self, x):
        H, _ = self.lstm(x)  # [batch_size, seq_len, hidden_size * 2]
        
        out = self.attention(H)  # [batch_size, hidden_size * 2]

        
        out = F.relu(self.fc1(out))
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc(out)
        
        return out






















    