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


class PLMs_DataProcessor:
    """数据预处理处理器"""
    def __init__(self, tokenizer, config: ModelConfig):

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.tokenizer = tokenizer
        self.config = config

    def _load_data(self, data_path) -> None:
        """加载数据集"""
        model_type = self.config.pretrained_model_type
        try:
            data_df = pd.read_csv( data_path )
            # 定义序列变换函数
            def transform_sequence(row):
                sequence = str(row['sequence'])
                sequence = row['sequence'].upper()
                seq_type = row['sequence_type']            
                # 处理 "protein" 模型：需要蛋白质序列
                if model_type == "protein":
                    if seq_type == "dna":
                        return str(Seq(sequence).translate()).replace("*", "")
                    elif seq_type == "rna":
                        dna_seq = sequence.replace("U", "T")
                        return str(Seq(dna_seq).translate()).replace("*", "")
                    elif seq_type == "protein":
                        return sequence
                    else:
                        raise ValueError(f"不支持的 sequence_type: {seq_type} 用于模型 {model_type}")                
                elif model_type == "codon":
                    if seq_type == "dna":
                        if "mistral-codon" in self.config.pretrained_model.lower():
                            return sequence
                        elif "cdsbert" in self.config.pretrained_model.lower():
                            sequence = str(sequence).replace("T", "U")
                            sequence = self._translate_rna_to_single_letter_codon(sequence)
                            return sequence
                        elif "bioobang" in self.config.pretrained_model.lower():
                            return sequence
                        else:
                            return str(sequence).replace("T", "U")
                    elif seq_type == "rna":
                        return sequence
                    elif seq_type == "protein":
                        raise ValueError(f"需要逆向翻译（密码子优化）: {model_type}")
                else:
                    raise ValueError(f"未知的 pretrained_model_type: {model_type}")
            # 对每一行应用变换   
            data_df['sequence'] = data_df.apply(transform_sequence, axis=1)
            return data_df
        except FileNotFoundError:
            print("Error")

    def _preprocess_sequence(self, sequence: str) -> str:

        # CaLM的特殊处理
        if "calm" in self.config.pretrained_model.lower():
            sequence = self._codon_tokenize(sequence)
            
        # T5系列的特殊处理
        if "t5" in self.config.pretrained_model.lower():
            sequence = re.sub(r'[OBUZJ]', 'X', sequence)
            sequence = " ".join(list(sequence))

        return sequence

    def _codon_tokenize(self, sequence: str) -> str:
        """CodonBERT专用分词处理：3-mer分词"""
        kmer_len = 3
        stride = 3
        return " ".join(
            sequence[i:i+kmer_len] 
            for i in range(0, len(sequence) - kmer_len + 1, stride)
        )
    
    def _create_dataset(self, df: pd.DataFrame) -> Dataset:
        """创建训练数据集"""
        processed_seqs = [self._preprocess_sequence(seq) for seq in df['sequence']]

        if "t5" in self.config.pretrained_model.lower():
            tokenized = self.tokenizer(processed_seqs,
                                        max_length=self.config.max_seq_length, # 1024
                                        padding=True,
                                        truncation=True)       
            return Dataset.from_dict({**tokenized, "id":df["name"], "sequence":df["sequence"], "labels": df['label']})
        
        elif "calm" in self.config.pretrained_model.lower():
            tokenized = self.tokenizer(
                    processed_seqs, 
                    max_length=self.config.max_seq_length,
                    padding=True,          # 填充到相同长度
                    truncation=True,        # 截断到最大长度
                )
            return Dataset.from_dict({**tokenized, "id":df["name"], "sequence":df["sequence"], "labels": df['label']})

    def _translate_rna_to_single_letter_codon(self, rna_sequence: str) -> str:
        # 按照每3个碱基分割成密码子
        codons = [rna_sequence[i:i+3] for i in range(0, len(rna_sequence), 3)]
        
        # 查找对应的氨基酸
        aa_sequence = [self.codon_to_aa.get(codon, '?') for codon in codons]
        
        return ''.join(aa_sequence)

    def _split_data(self, df, test_size=0.2, random_state = 42):
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
    
    def _load_embedding_from_pickle(self, file_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """加载数据"""
        try:
            df = pd.read_pickle(file_path)
            features = pd.DataFrame(df.pop('embedding').to_list(), index=df.index)
            labels = df["labels"]
            return features, labels
        except Exception as e:

            print(f"加载数据失败: {str(e)}")
            raise

    def _preprocess_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """预处理数据"""
        try:
            X_scaled = self.scaler.fit_transform(X)
            if np.all(y == y[0]):  # 如果所有值都相同
                y_scaled = y.values  # 不缩放
            else:
                y_scaled = self.scaler.fit_transform(y.values.reshape(-1, 1)).ravel()
            # y_scaled = self.scaler.fit_transform(y.values.reshape(-1, 1)).ravel()
            y_scaled = y_scaled.astype(int)
            return X_scaled, y_scaled
        except Exception as e:
            print(f"数据预处理失败: {str(e)}")
            raise
    
class UniM_DataProcessor:
    """数据预处理处理器"""
    def __init__(self, calm_tokenizer, t5_tokenizer):

        self.calm_tokenizer = calm_tokenizer
        self.t5_tokenizer = t5_tokenizer
     
    def _load_data(self, data_path) -> None:
        """加载数据集"""
        data_df = pd.read_csv( data_path )
        data_df['calm_sequence'] = data_df.sequence.apply( lambda x: str(x).replace("T", "U") )
        data_df['t5_sequence'] = data_df.sequence.apply( lambda x:  str(Seq(x).translate()).replace("*", "")  )
        data_df = data_df.drop(columns=["sequence"])

        return data_df

    def _codon_tokenize(self, sequence: str) -> str:
        """CodonBERT专用分词处理：3-mer分词"""
        kmer_len = 3
        stride = 3
        return " ".join(
            sequence[i:i+kmer_len] 
            for i in range(0, len(sequence) - kmer_len + 1, stride)
        )

    
    def _create_dataset(self, calm_df: pd.DataFrame, t5_df: pd.DataFrame) -> Dataset:

        calm_processed_seqs = [self.self._codon_tokenize(seq) for seq in calm_df['sequence']]
        t5_processed_seqs = [ " ".join(list(  re.sub(r'[OBUZJ]', 'X', seq)  ))   for seq in t5_df['sequence']]

        # 为CaLM tokenizer处理
        calm_tokenized = self.calm_tokenizer(
            calm_processed_seqs, 
            max_length=self.config.calm_max_length,  # 例如512
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # 为T5 tokenizer处理
        t5_tokenized = self.t5_tokenizer(
            t5_processed_seqs,
            max_length=self.config.t5_max_length,   # 例如512
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # 创建双模态数据集
        dataset_dict = {
            # CaLM相关字段
            "calm_input_ids": calm_tokenized["input_ids"],
            "calm_attention_mask": calm_tokenized["attention_mask"],
            
            # T5相关字段
            "t5_input_ids": t5_tokenized["input_ids"],
            "t5_attention_mask": t5_tokenized["attention_mask"],
            
            # 原始数据
            "labels": calm_df['label']
        }
        
        return Dataset.from_dict(dataset_dict)

class Load_PLMs:
    """序列分类模型处理器"""
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化模型和分词器，根据模型名称选择加载方式
        if "t5" in config.pretrained_model.lower():
            self.model, self.tokenizer = self._load_t5_model()

        elif "calm" in config.pretrained_model.lower():
            self.model, self.tokenizer = self._load_calm_model()
              
    def _load_calm_model(self):
        if self.config.task == "do_finetune":
            """加载calm密码子语言模型"""
            # 注册配置
            AutoConfig.register("calm", CaLMConfig)
            CONFIG_MAPPING["calm"] = CaLMConfig

            # 注册模型
            AutoModel.register(CaLMConfig, CaLMModel)
            MODEL_MAPPING[CaLMConfig] = CaLMModel

            #注册分词器
            AutoTokenizer.register(CaLMConfig, CaLMTokenizer)
            
            # 加载配置、模型、分词器
            config, model, tokenizer = AutoConfig.from_pretrained(self.config.pretrained_model), AutoModel.from_pretrained(self.config.pretrained_model), AutoTokenizer.from_pretrained(self.config.pretrained_model)
            
            #加载分类模型
            class_model = CaLMForSequenceClassification(config)
            class_model.calm = model

            # Delete the checkpoint model
            model = class_model
            del class_model

            # Print number of trainable parameters
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            print("calm_Classfier\nTrainable Parameter: "+ str(params))

            # 添加LoRA适配器
            if self.config.finetune_type == "lora":
                peft_config = LoraConfig(
                    r=self.config.lora_rank, 
                    lora_alpha=1,
                    bias="lora_only",
                    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]
                )
                model = inject_adapter_in_model(peft_config, model)
    
            # Unfreeze the prediction head
            for (param_name, param) in model.classifier.named_parameters():
                param.requires_grad = True

            # Print trainable Parameter    
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            print("calm_LoRA_Classfier\nTrainable Parameter: "+ str(params) + "\n") 
        
        return model, tokenizer 

    def _load_t5_model(self):

        if self.config.task == "do_finetune":
            """加载T5系列模型"""
            model = T5EncoderModel.from_pretrained(self.config.pretrained_model)            
            tokenizer = T5Tokenizer.from_pretrained(self.config.pretrained_model) 
            
            # Create new Classifier model with PT5 dimensions
            class_config = T5ClassConfig(num_labels=self.config.num_labels)
            class_model = T5EncoderForSimpleSequenceClassification(model.config, class_config)

            # Set encoder and embedding weights to checkpoint weights
            class_model.shared = torch.nn.Embedding.from_pretrained(model.shared.weight.clone(), freeze=False)
            class_model.encoder = model.encoder

            # Delete the checkpoint model
            model = class_model
            del class_model

            # Print number of trainable parameters
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            print("T5_Classfier\nTrainable Parameter: "+ str(params))   

            # 添加LoRA适配器
            if self.config.finetune_type == "lora":
                peft_config = LoraConfig(
                    r=self.config.lora_rank,
                    lora_alpha=1,
                    bias="lora_only",
                    target_modules=["q","k","v","o"]
                )
                model = inject_adapter_in_model(peft_config, model)
            
            # Unfreeze the prediction head
            for (param_name, param) in model.classifier.named_parameters():
                param.requires_grad = True
            # Print trainable Parameter    
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            print("T5_Classfier\nTrainable Parameter: "+ str(params) + "\n") 


        return model, tokenizer