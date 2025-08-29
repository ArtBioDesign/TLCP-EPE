# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Config(object):
    """配置参数"""
    def __init__(self):
        self.model_name = 'Transformer_MLP'
        self.class_list = list(range(2))             
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
        self.dropout = 0.1                                          
        self.num_classes = 2                        
        self.embed = 1792 
        self.hidden_size = 1792     # Transformer d_model
        self.num_heads = 8          # 注意力头数
        self.num_layers = 5         # Transformer 层数
        self.hidden_size2 = 64
        self.max_seq_len = 1792      # 最大序列长度

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(config.embed, config.max_seq_len, config.dropout)
        
        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed,
            nhead=config.num_heads,
            dim_feedforward=config.embed * 4,  # 通常设置为 d_model 的 4 倍
            dropout=config.dropout,
            batch_first=True,
            activation='gelu'  # 使用 GELU 激活函数
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config.num_layers
        )
        
        # 注意力池化层
        self.attention_pool = nn.Linear(config.embed, 1)
        
        # MLP 分类器
        self.fc1 = nn.Linear(config.embed, config.hidden_size2)
        self.bn1 = nn.BatchNorm1d(config.hidden_size2)
        self.dropout = nn.Dropout(config.dropout) 
        self.fc = nn.Linear(config.hidden_size2, config.num_classes)
        
    def forward(self, x):
        # x: [batch_size, seq_len, embed_dim]
        batch_size, seq_len, embed_dim = x.shape
        
        # 添加位置编码
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)  # 保持 batch_first
        
        # Transformer 编码
        # src_key_padding_mask 可以用于处理变长序列（如果需要）
        transformer_out = self.transformer_encoder(x)  # [batch_size, seq_len, embed_dim]
        
        # 注意力池化 - 聚合序列信息
        attention_weights = self.attention_pool(transformer_out)  # [batch_size, seq_len, 1]
        attention_weights = F.softmax(attention_weights, dim=1)   # [batch_size, seq_len, 1]
        
        # 加权平均
        out = torch.sum(transformer_out * attention_weights, dim=1)  # [batch_size, embed_dim]
        
        # MLP 分类
        out = F.gelu(self.fc1(out))  # 使用 GELU 激活函数
        out = self.bn1(out)
        out = self.dropout(out)
        out = self.fc(out)
        
        return out
