import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


class Config(object):
    """配置参数"""
    def __init__(self):
        self.model_name = 'LowRankTensorFusion'
        self.class_list = list(range(2))             
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
        self.dropout = 0.1                                          
        self.num_classes = 2                        
        self.img_dim = 1024       # 图像特征维度
        self.text_dim = 768       # 文本特征维度
        self.ir_dim = 768         # 中间表示维度
        self.rank = 8             # 低秩张量的秩
        self.hidden_size2 = 64    # MLP隐藏层大小


class LowRankTensorFusion(nn.Module):
    """
    Implementation of Low-Rank Tensor Fusion.
    """

    def __init__(
        self, input_dims=[1024, 768], ir_dim=256, output_dim=2, rank=2, flatten=True
    ):
        """
        Initialize LowRankTensorFusion object.

        :param input_dims: list or tuple of integers indicating input dimensions of the modalities
        :param ir_dim: intermediate representation dimension
        :param output_dim: output dimension
        :param rank: a hyperparameter of LRTF. See link above for details
        :param flatten: Boolean to dictate if output should be flattened or not. Default: True
        """
        super(LowRankTensorFusion, self).__init__()

        # dimensions are specified in the order of image, text
        self.input_dims = input_dims
        self.ir_dim = ir_dim
        self.output_dim = output_dim
        self.rank = rank
        self.flatten = flatten

        # low-rank factors
        self.factors = nn.ParameterList()
        for input_dim in input_dims:
            factor = nn.Parameter(
                torch.Tensor(self.rank, input_dim + 1, self.ir_dim)
            )
            nn.init.xavier_normal_(factor)
            self.factors.append(factor)

        self.fusion_weights = nn.Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = nn.Parameter(torch.Tensor(1, self.ir_dim))
        
        # init the fusion weights
        nn.init.xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self, img_feat, text_feat):
        """
        Forward Pass of Low-Rank TensorFusion.

        :param img_feat: image features [B, img_dim]
        :param text_feat: text features [B, text_dim]
        """
        modalities = [img_feat, text_feat]
        batch_size = modalities[0].shape[0]
        
        # next we perform low-rank multimodal fusion
        # here is a more efficient implementation than the one the paper describes
        # basically swapping the order of summation and elementwise product
        fused_tensor = 1
        for modality, factor in zip(modalities, self.factors):
            ones = Variable(
                torch.ones(batch_size, 1).type(modality.dtype), requires_grad=False
            ).to(modality.device)
            if self.flatten:
                modality_withones = torch.cat(
                    (ones, torch.flatten(modality, start_dim=1)), dim=1
                )
            else:
                modality_withones = torch.cat((ones, modality), dim=1)
            modality_factor = torch.matmul(modality_withones, factor)
            fused_tensor = fused_tensor * modality_factor

        output = (
            torch.matmul(self.fusion_weights, fused_tensor.permute(1, 0, 2)).squeeze()
            + self.fusion_bias
        )
        output = output.view(-1, self.ir_dim)
        return output


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        
        # 使用低秩张量融合替代原来的交叉注意力融合
        self.fusion = LowRankTensorFusion(
            input_dims=[config.img_dim, config.text_dim],
            ir_dim=config.ir_dim,
            output_dim=config.num_classes,
            rank=config.rank,
            flatten=True
        )

        # 分类头
        self.cls_head = nn.Sequential(
            nn.Linear(config.ir_dim, config.hidden_size2),
            nn.BatchNorm1d(config.hidden_size2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size2, config.num_classes)
        )

    def forward(self, x):
        # x: [B, 1, 1792] 或 [B, 1792]
        x = x.view(x.size(0), -1)  # [B, 1792]
        
        text_feat = x[:, :self.config.text_dim]   # [B, 768]
        img_feat = x[:, self.config.text_dim:]    # [B, 1024]
        
        # 通过低秩张量融合
        fused = self.fusion(img_feat, text_feat)  # [B, ir_dim]
        
        # 分类
        output = self.cls_head(fused)
        return output