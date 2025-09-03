import os 
import torch
from torch.utils.data import random_split
import pandas as pd
import pandas as pd
import numpy as np
import os, sys, subprocess
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn import metrics
import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler
from typing import List, Tuple, Dict, Any, Optional
import logging
from torch.utils.data import random_split
from datasets import Dataset
from config import *
from Bio.Seq import Seq
import os,sys
import re
from peft import LoraConfig, inject_adapter_in_model
from T5E import *

from transformers import (
    T5EncoderModel, 
    T5Tokenizer,
    AutoTokenizer,
    AutoModel,
    AutoConfig
)
current_dir = os.path.dirname(os.path.abspath(__file__))
pretrain_codon_path = os.path.join(current_dir, "pretrain_codon")
sys.path.append(pretrain_codon_path)

from hugCalm import CaLMConfig, CaLMModel, CaLMTokenizer
from calmcls import CaLMForSequenceClassification
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_MAPPING

from torch.utils.data import DataLoader, Dataset, random_split

def split_dataset(dataset, ratio=0.8):
    # 计算数据集大小
    dataset_size = len(dataset)
    train_size = int(ratio * dataset_size)
    val_size = dataset_size - train_size

    # 设置随机种子，保证每次划分的数据集相同
    torch.manual_seed(42)
    # 按照 8:1:1 划分数据集
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset

def get_matricx( n, column1 ,column2, column3 ):

    temp = n.copy()
    temp = temp.set_index(column1)
    n = pd.DataFrame(temp.pop(column2).to_list(), index=temp.index)
    n[column3] = temp[column3]

    return n   

class EarlyStopping:
    """早停机制"""
    def __init__(
        self, 
        patience=10, 
        verbose=False, 
        delta=0, 
        path='checkpoint.pt',
        trace_func=print
    ):
        """
        Args:
            patience (int): 多少个epoch没有改进后停止训练
            verbose (bool): 是否打印早停信息
            delta (float): 被认为是改进的最小变化
            path (str): 模型保存路径
            trace_func (function): 打印函数
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        """
        Args:
            val_loss (float): 当前验证损失
            model (torch.nn.Module): 当前模型
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """保存模型检查点"""
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        # 确保检查点目录存在
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class BiolmDataSet(Dataset):
    def __init__(self, X, y):
        self.x = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __getitem__(self, index):

        code = self.x[index].unsqueeze(0)
        label = self.y[index]
        return code, label

    def __len__(self):
        return len(self.x)
    


class ModelEvaluator:
    """模型评估器"""
    def __init__(self):
        self.name = "metric" 

    def evaluate_model(self, y_pred, y_test, y_pred_prob_fold, task_type="bc"):
            
        metrics_dict = {}

        true_labels = y_test
        pred_labels = y_pred
        if task_type == "bc":
            metrics_dict.update({
                    "AUC": None if set(y_test) == 1  else metrics.roc_auc_score( true_labels, y_pred_prob_fold ),
                    "ACC": metrics.accuracy_score(true_labels, pred_labels),
                    "Precision": metrics.precision_score(true_labels, pred_labels),
                    "Recall": metrics.recall_score(true_labels, pred_labels),
                    "F1": metrics.f1_score(true_labels, pred_labels),
                    "MCC": metrics.matthews_corrcoef(true_labels, pred_labels),
                    # "sensitivity": metrics.recall_score(true_labels, pred_labels),
                    # "specificity": self._specificity(true_labels, pred_labels)
                })
        elif task_type == "mcc":
            
            metrics_dict.update({
                    "AUC": metrics.roc_auc_score( true_labels, y_pred_prob_fold, multi_class='ovr', average='macro' ),
                    "ACC": metrics.accuracy_score(true_labels, pred_labels),
                    "Precision": metrics.precision_score(true_labels, pred_labels, average="macro"),
                    "Recall": metrics.recall_score( true_labels, pred_labels, average="macro" ),
                    "F1": metrics.f1_score(true_labels, pred_labels, average="macro"),
                    "MCC": metrics.matthews_corrcoef(true_labels, pred_labels)
                })
        
        return metrics_dict
    
class DLMs_DataProcessor:
    """数据处理器"""
    def __init__(self):

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def load_data(self, file_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """加载数据"""
        try:
            df = pd.read_pickle(file_path)
            features = pd.DataFrame(df.pop('embedding').to_list(), index=df.index)
            labels = df["labels"]
            return features, labels
        except Exception as e:

            print(f"加载数据失败: {str(e)}")
            raise

    def preprocess_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """预处理数据"""
        try:
            X_scaled = self.scaler.fit_transform(X)
            if np.all(y == y[0]):  # 如果所有值都相同
                y_scaled = y.values  # 不缩放
            else:
                y_scaled = self.scaler.fit_transform(y.values.reshape(-1, 1)).ravel()
            # y_scaled = self.scaler.fit_transform(y.values.reshape(-1, 1)).ravel()
            return X_scaled, y_scaled
        except Exception as e:
            print(f"数据预处理失败: {str(e)}")
            raise
    
    def split_data(self, df, test_size=0.2, random_state = 42):
        """ 数据划分 """
        try:
            X = df.drop(['label'], axis=1).values
            y = df['label'].values

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )

            print("训练集标签分布：", dict(zip(*np.unique(y_train, return_counts=True))))
            print("测试集标签分布：", dict(zip(*np.unique(y_test, return_counts=True))))
            
            return X_train, X_test, y_train, y_test
        
        except Exception as e:
            print(f"数据划分失败: {str(e)}")
            raise




