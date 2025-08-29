import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
import sys
sys.path.append( "/hpcfs/fhome/yangchh/workdir/self/TLCP-EPE/DLs" )
from utils import Load_PLMs
from config import T5_PLM_CONFIG, CALM_PLM_CONFIG


class Config(object):
    """配置参数"""
    def __init__(self):
        self.model_name = 'Attention_MLP'
        self.class_list = list(range(2))             
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
        self.dropout = 0.1                                          
        self.num_classes = 2                        
        self.embed = 1024 
        self.hidden_size = 1792   # Attention层的隐藏大小                                   
        self.hidden_size2 = 64   # MLP层的隐藏大小
        self.t5_encoder = ""
        self.calm_encoder= ""



class BANLayer(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=0.2, k=3):
        super(BANLayer, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout)
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout)
        # self.dropout = nn.Dropout(dropout[1])
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

        self.bn = nn.BatchNorm1d(h_dim)

    def attention_pooling(self, v, q, att_map):
        fusion_logits = torch.einsum('bvk,bvq,bqk->bk', (v, att_map, q))
        if 1 < self.k:
            fusion_logits = fusion_logits.unsqueeze(1)  # b x 1 x d
            fusion_logits = self.p_net(fusion_logits).squeeze(1) * self.k  # sum-pooling
        return fusion_logits

    def forward(self, v, q, softmax=False):
        v_num = v.size(1)
        q_num = q.size(1)
        if self.h_out <= self.c:
            v_ = self.v_net(v)
            q_ = self.q_net(q)
            att_maps = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
        else:
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q
            att_maps = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # b x v x q x h_out
            att_maps = att_maps.transpose(2, 3).transpose(1, 2)  # b x h_out x v x q
        if softmax:
            p = nn.functional.softmax(att_maps.view(-1, self.h_out, v_num * q_num), 2)
            att_maps = p.view(-1, self.h_out, v_num, q_num)
        logits = self.attention_pooling(v_, q_, att_maps[:, 0, :, :])
        for i in range(1, self.h_out):
            logits_i = self.attention_pooling(v_, q_, att_maps[:, i, :, :])
            logits += logits_i
        logits = self.bn(logits)
        return logits, att_maps

class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    Modified from https://github.com/jnhwkim/ban-vqa/blob/master/fc.py
    """

    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class BCNet(nn.Module):
    """Simple class for non-linear bilinear connect network
    Modified from https://github.com/jnhwkim/ban-vqa/blob/master/bc.py
    """

    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=[.2, .5], k=3):
        super(BCNet, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim;
        self.q_dim = q_dim
        self.h_dim = h_dim;
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.dropout = nn.Dropout(dropout[1])  # attention
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if None == h_out:
            pass
        elif h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

    def forward(self, v, q):
        if None == self.h_out:
            v_ = self.v_net(v)
            q_ = self.q_net(q)
            logits = torch.einsum('bvk,bqk->bvqk', (v_, q_))
            return logits

        # low-rank bilinear pooling using einsum
        elif self.h_out <= self.c:
            v_ = self.dropout(self.v_net(v))
            q_ = self.q_net(q)
            logits = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
            return logits  # b x h_out x v x q

        # batch outer product, linear projection
        # memory efficient but slow computation
        else:
            v_ = self.dropout(self.v_net(v)).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q
            logits = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # b x v x q x h_out
            return logits.transpose(2, 3).transpose(1, 2)  # b x h_out x v x q

    def forward_with_weights(self, v, q, w):
        v_ = self.v_net(v)  # b x v x d
        q_ = self.q_net(q)  # b x q x d
        logits = torch.einsum('bvk,bvq,bqk->bk', (v_, w, q_))
        if 1 < self.k:
            logits = logits.unsqueeze(1)  # b x 1 x d
            logits = self.p_net(logits).squeeze(1) * self.k  # sum-pooling
        return logits

class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        return x

class CaLM_encoder(nn.Module):
    def __init__(self, config):
        super(CaLM_encoder, self).__init__()
        self.calm_encoder = config.calm_encoder[0]
        self.calm_encoder = self.calm_encoder.to(config.device)
        # 冻结参数
        for param in self.calm_encoder.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask=None):

        result = self.calm_encoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        repr_layers=[12]
                        )
        repr_ = result["hidden_states"]

        return repr_

class T5_encoder(nn.Module):
    def __init__(self, config):
        super(T5_encoder, self).__init__()
        self.t5_encoder = config.t5_encoder
        self.t5_encoder = self.t5_encoder.to(config.device)

        for param in self.t5_encoder.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        result = self.t5_encoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True
                    )
        repr_ = result["hidden_states"][-1]

        return repr_
        
class DimensionAlignment(nn.Module):
    """维度对齐网络，将不同维度的特征对齐到统一维度"""
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super(DimensionAlignment, self).__init__()
        self.projection = nn.Sequential(
            weight_norm(nn.Linear(input_dim, output_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout),
            weight_norm(nn.Linear(output_dim, output_dim), dim=None),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x):
        return self.projection(x)



class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        mlp_in_dim =256
        ban_heads=2
        mlp_hidden_dim = 512
        mlp_out_dim =128

        self.calm_encoder = CaLM_encoder(config ) #batch, seqlength, 768
        self.t5_encoder = T5_encoder(config)   #batch, seqlength, 1024

        self.calm_alignment = DimensionAlignment(768, 512)  
        self.t5_alignment = DimensionAlignment(1024, 512)   

        self.bcn = weight_norm(
                            BANLayer(v_dim=512, q_dim=512, h_dim=mlp_in_dim, h_out=ban_heads),
                            name='h_mat', 
                            dim=None
                            )
        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim)
        self.classifier = nn.Linear(mlp_out_dim, 2)


    def forward(self, batch):

        calm_input_ids = batch['calm_input_ids']
        calm_attention_mask = batch['calm_attention_mask']
        t5_input_ids = batch['t5_input_ids']
        t5_attention_mask = batch['t5_attention_mask']

        calm_outputs = self.calm_encoder(
                input_ids=calm_input_ids, 
                attention_mask=calm_attention_mask
            )

        t5_outputs = self.t5_encoder(
                input_ids=t5_input_ids, 
                attention_mask=t5_attention_mask
            )

        calm_aligned = self.calm_alignment(calm_outputs)
        t5_aligned = self.t5_alignment(t5_outputs)

        fused_features, attention_weights = self.bcn(calm_aligned, t5_aligned)

        score = self.mlp_classifier( fused_features )

        logits_clsf = self.classifier(score)

        return logits_clsf

        # f, att = self.bcn(v_f, v_p)
        # score = self.mlp_classifier(f)
        # logits_clsf = self.classifier(score)

        # return logits_clsf, f