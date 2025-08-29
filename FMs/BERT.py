#import dependencies
import os.path
# # os.chdir("set working path here")
# os.chdir("/hpcfs/fhome/yangchh/ai/data-repo_plm-finetune-eval/training data")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader
import re
import numpy as np
import pandas as pd 
import copy
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import BertConfig, BertForSequenceClassification
from transformers import BertModel, BertTokenizer


class BERTClassConfig:
    def __init__(self, dropout=0, num_labels=1):
        self.dropout_rate = dropout
        self.num_labels = num_labels

class BERTClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, class_config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.dropout = nn.Dropout(class_config.dropout_rate)
        self.out_proj = nn.Linear(config.hidden_size, class_config.num_labels, bias=True)

    def forward(self, hidden_states):

        hidden_states =  torch.mean(hidden_states,dim=1)  # avg embedding 

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states

class BertForSimpleSequenceClassification(nn.Module):  
    def __init__(self, config: BertConfig, class_config):
        super(BertForSimpleSequenceClassification, self).__init__()
        self.num_labels = class_config.num_labels
        self.config = config

        self.bert = BertModel(config)

        # self.bert = BertModel.from_pretrained(class_config.modelname)

        self.dropout = nn.Dropout(class_config.dropout_rate) 
        self.classifier = BERTClassificationHead(config, class_config)

    #     # Initialize weights and apply final processing
    #     self._init_weights()

    # def _init_weights(self):
    #     """Initialize the weights of the classifier"""
    #     nn.init.xavier_normal_(self.classifier.weight)
    #     if self.classifier.bias is not None:
    #         nn.init.zeros_(self.classifier.bias)


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
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
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )