from transformers import PreTrainedModel
import torch.nn as nn
import torch
from hugCalm import CaLMModel, CaLMConfig
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer
from transformers import AutoModel, AutoConfig, AutoTokenizer
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from sklearn.svm import SVC
import numpy as np

from transformers import PreTrainedModel
import torch.nn as nn
import torch
from hugCalm import CaLMModel, CaLMConfig
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer
from transformers import AutoModel, AutoConfig, AutoTokenizer
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from sklearn.svm import SVC
import numpy as np


class CaLMClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.embed_dim, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.out_proj = nn.Linear(config.embed_dim, config.num_labels)

    def forward(self, hidden_states):

        hidden_states =  torch.mean(hidden_states,dim=1)  # avg embedding

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states

class CaLMForSequenceClassification(PreTrainedModel):
    config_class = CaLMConfig
    # base_model_prefix = "calm"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels  # 分类标签数量
        self.config = config

        self.calm = CaLMModel(config)  # 基础 CaLM 模型

        self.dropout = nn.Dropout( config.dropout_rate ) 

        self.classifier = CaLMClassificationHead(config)
        

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        labels=None,
        return_dict=None,
        **kwargs
    ):
        # 获取 CaLMModel 的输出
        outputs = self.calm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            return_dict=True
        )

        # 提取最后一层隐藏状态
        hidden_states = outputs["representations"][12]

        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
            
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states= outputs["representations"][12], # outputs["representations"][12],
            attentions=None # outputs["representations"],
        )



