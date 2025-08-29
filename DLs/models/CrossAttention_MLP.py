import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):
    """配置参数"""
    def __init__(self):
        self.model_name = 'CrossAttention_MLP'
        self.class_list = list(range(2))             
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
        self.dropout = 0.1                                          
        self.num_classes = 2                        
        self.img_dim = 1024       # 图像特征维度
        self.text_dim = 768       # 文本特征维度
        self.d_model = 512        # 统一投影到同一维度
        self.nhead = 8            # 注意力头数
        self.num_layers = 2       # 多层交叉注意力
        self.hidden_size2 = 64    # MLP隐藏层大小


class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout=0.1):
        super(CrossAttentionFusion, self).__init__()
        self.d_model = d_model
        
        # 投影层，将不同维度映射到统一维度
        self.img_proj = nn.Linear(1024, d_model)
        self.text_proj = nn.Linear(768, d_model)
        
        # 真正的Cross-Attention层
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # 可选：后续的Transformer层来进一步处理融合后的特征
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 全局池化层
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, img_feat, text_feat):
        # 添加序列维度 [B, D] -> [B, 1, D]
        if len(img_feat.shape) == 2:
            img_feat = img_feat.unsqueeze(1)  # [B, 1, 1024]
        if len(text_feat.shape) == 2:
            text_feat = text_feat.unsqueeze(1)  # [B, 1, 768]
            
        # 投影到统一维度
        img_feat_proj = self.img_proj(img_feat)      # [B, 1, d_model]
        text_feat_proj = self.text_proj(text_feat)   # [B, 1, d_model]
        
        # 方法1: 双向Cross-Attention (文本查询图像，图像查询文本)
        # 文本作为Query，图像作为Key和Value
        text_to_img, _ = self.cross_attn(text_feat_proj, img_feat_proj, img_feat_proj)
        # 图像作为Query，文本作为Key和Value  
        img_to_text, _ = self.cross_attn(img_feat_proj, text_feat_proj, text_feat_proj)
        
        # 融合两种注意力结果
        fused_feat = torch.cat([text_to_img, img_to_text], dim=1)  # [B, 2, d_model]
        
        # 进一步通过Transformer处理
        out = self.transformer_encoder(fused_feat)  # [B, 2, d_model]
        
        # 全局池化得到最终特征
        out = out.transpose(1, 2)  # [B, d_model, 2]
        out = self.global_pool(out)  # [B, d_model, 1]
        out = out.squeeze(-1)  # [B, d_model]
        
        return out


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.fusion = CrossAttentionFusion(
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dropout=config.dropout
        )

        self.fc1 = nn.Linear(config.d_model, config.hidden_size2)
        self.bn1 = nn.BatchNorm1d(config.hidden_size2)
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.hidden_size2, config.num_classes)

    def forward(self, x):
        # x: [B, 1, 1792] 或 [B, 1792]
        if len(x.shape) == 3:
            x = x.squeeze(1)  # [B, 1792]
        
        text_feat = x[:, :768]   # [B, 768]
        img_feat = x[:, 768:]    # [B, 1024]
        
        fused = self.fusion(img_feat, text_feat)  # [B, d_model]

        out = F.relu(self.fc1(fused))
        out = self.bn1(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out