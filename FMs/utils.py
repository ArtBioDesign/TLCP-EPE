import pandas as pd
import numpy as np
import sys
import torch
from dataclasses import dataclass
from peft import LoraConfig, inject_adapter_in_model
from datasets import Dataset
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_MAPPING

sys.path.append("/hpcfs/fhome/yangchh/workdir/self/TLCP-EPE/method/BiooBang")
from model.modeling_UniBioseq import UniBioseqForEmbedding, UBSLMForSequenceClassification, UniBioseqForSequenceClassification_bidirectional, UniBioseqForSequenceClassification_convbert
from model.tokenization_UniBioseq import UBSLMTokenizer
from transformers import (
    T5EncoderModel, 
    T5Tokenizer,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModel,
    BertForSequenceClassification,
    BertTokenizer,
    BertConfig,
    BertModel,
    EsmModel,
    BertForPreTraining,
    AutoModel,
    AutoConfig
)
from Bio.Seq import Seq
import os,sys
import re

current_dir = os.path.dirname(os.path.abspath(__file__))
pretrain_codon_path = os.path.join(current_dir, "pretrain_codon")
sys.path.append(pretrain_codon_path)

from hugCalm import CaLMConfig, CaLMModel, CaLMTokenizer
from calmcls import CaLMForSequenceClassification
from T5E import *
from BERT import *
from MCCodon import *

pretrain_cdsFM_path = os.path.join(current_dir, "cdsFM")
sys.path.append(pretrain_cdsFM_path)
from cdsFM.cdsfm import AutoEnCodon

@dataclass
class ModelConfig:
    """模型训练配置参数"""
    pretrained_model: str = "lhallee/CodonBERT"
    data_path: str = "./data/5fold-1"
    num_labels: int = 2
    batch_size: int = 1
    gradient_accumulation: int = 8 
    epochs: int = 5
    learning_rate: float = 0.000005
    seed: int = 42
    mixed_precision: bool = False 
    max_seq_length: int = 1024
    lora_rank: int = 4
    task: str = "do_finetune"
    finetuned_params_name: str ="finetuned_params.pth"
    train: bool = True
    file_name: str = "finetune_FMs_data"
    save_model_path: str = ""
    pretrained_model_type: str = "protein"
    finetune_type: str="lora"

class SequenceClassifier:
    """序列分类模型处理器"""
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化模型和分词器，根据模型名称选择加载方式
        if "t5" in config.pretrained_model.lower() or "ankh" in config.pretrained_model.lower():
            self.model, self.tokenizer = self._load_t5_model()
        elif "esm" in config.pretrained_model.lower():
            self.model, self.tokenizer = self._load_esm_model()
        elif "prot_bert" in config.pretrained_model.lower():
            self.model, self.tokenizer = self._load_protbert_model()

        elif "calm" in config.pretrained_model.lower():
            self.model, self.tokenizer = self._load_calm_model()
        elif "codonbert" in config.pretrained_model.lower():
            self.model, self.tokenizer = self._load_codonbert_model()
        elif "encodon" in config.pretrained_model.lower():
            self.model, self.tokenizer = self._load_cdsFms_model() 
        elif "mistral-codon" in config.pretrained_model.lower():
            self.model, self.tokenizer = self._load_mcodon_model()
        elif "cdsbert" in config.pretrained_model.lower():
            self.model, self.tokenizer = self._load_cdsbert_model()
        elif "bioobang" in config.pretrained_model.lower():
            self.model, self.tokenizer = self._load_bioobang_model()               

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

            elif self.config.finetune_type == "full":
    
                for (param_name, param) in model.named_parameters(): 
                    param.requires_grad = True
                print("calm_full_Classfier\nTrainable Parameter:") 

            elif self.config.finetune_type == "deep":

                for (param_name, param) in model.named_parameters(): 
                    param.requires_grad = False
                for (param_name, param) in model.calm.layers[10].named_parameters(): 
                    param.requires_grad = True
                for (param_name, param) in model.calm.layers[11].named_parameters(): 
                    param.requires_grad = True
                print("calm_deep_Classfier\nTrainable Parameter:") 
                
            elif self.config.finetune_type == "shallow":
                
                for (param_name, param) in model.named_parameters(): 
                    param.requires_grad = False
                for (param_name, param) in model.calm.layers[0].named_parameters(): 
                    param.requires_grad = True
                for (param_name, param) in model.calm.layers[1].named_parameters(): 
                    param.requires_grad = True
                print("calm_shallow_Classfier\nTrainable Parameter:")
                            
            else:
                # """Freezes all parameters in the model"""
                print("calm_freeze_Classfier\nTrainable Parameter:") 
                for (param_name, param) in model.named_parameters(): 
                    param.requires_grad = False
    
            # Unfreeze the prediction head
            for (param_name, param) in model.classifier.named_parameters():
                param.requires_grad = True

            # Print trainable Parameter    
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            print("calm_LoRA_Classfier\nTrainable Parameter: "+ str(params) + "\n") 
            
        elif self.config.task == "do_embedding":
            # 注册配置 
            AutoConfig.register("calm", CaLMConfig)
            CONFIG_MAPPING["calm"] = CaLMConfig

            # 注册模型
            AutoModel.register(CaLMConfig, CaLMModel)
            MODEL_MAPPING[CaLMConfig] = CaLMModel

            #注册分词器
            AutoTokenizer.register(CaLMConfig, CaLMTokenizer)

            # #模型权重
            # model_path = "/hpcfs/fhome/yangchh/ai/finetuneFMs/data/sgc-rna/saved_models_huggingface_calm_sag/results/checkpoint-179"
            
            # 加载配置、模型、分词器
            # config, model, tokenizer = AutoConfig.from_pretrained(model_path), AutoModel.from_pretrained(model_path), AutoTokenizer.from_pretrained(model_path)   
            config, model, tokenizer = AutoConfig.from_pretrained(self.config.pretrained_model), AutoModel.from_pretrained(self.config.pretrained_model), AutoTokenizer.from_pretrained(self.config.pretrained_model)   
        
        return model, tokenizer 

    def _load_bioobang_model(self):

        model = UniBioseqForSequenceClassification_convbert.from_pretrained(self.config.pretrained_model)
        tokenizer = UBSLMTokenizer.from_pretrained(self.config.pretrained_model)

        # """Freezes all parameters in the model"""
        for (param_name, param) in model.named_parameters(): 
            param.requires_grad = False

        # Print number of trainable parameters
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("BiooBang_Classfier\nTrainable Parameter: "+ str(params))

        # Unfreeze the prediction head  
        for (param_name, param) in model.convbert.named_parameters():
            param.requires_grad = True
        
        for (param_name, param) in model.pooling.named_parameters():
            param.requires_grad = True
        
        for (param_name, param) in model.score.named_parameters():
            param.requires_grad = True

        # Print trainable Parameter          
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("BiooBang_Classfier\nTrainable Parameter: "+ str(params) + "\n")

        return model, tokenizer

    def _load_codonbert_model(self):

        if self.config.task == "do_finetune":
            config = BertConfig.from_pretrained(self.config.pretrained_model)

            # model = AutoModelForSequenceClassification.from_pretrained(
            #         self.config.pretrained_model,
            #         num_labels=self.config.num_labels
            #     )
            tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_model)
            class_config = BERTClassConfig(num_labels=self.config.num_labels)
            model = BertForSimpleSequenceClassification(config, class_config)


            # Print number of trainable parameters
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            print("codonbert_Classfier\nTrainable Parameter: "+ str(params))

            if self.config.finetune_type == "lora":
                # 添加LoRA适配器
                peft_config = LoraConfig(
                    r=self.config.lora_rank,
                    lora_alpha=1,
                    target_modules=["query", "key", "value", "dense"],
                    bias="lora_only"
                )
                model = inject_adapter_in_model(peft_config, model)
            else:
                # """Freezes all parameters in the model"""
                print("codonbert_freeze_Classfier\nTrainable Parameter:") 
                for (param_name, param) in model.named_parameters(): 
                    param.requires_grad = False

            # Unfreeze the prediction head
            for (param_name, param) in model.classifier.named_parameters():
                param.requires_grad = True  

            # Print trainable Parameter          
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            print("codonbert_LoRA_Classfier\nTrainable Parameter: "+ str(params) + "\n")
        
        elif self.config.task == "do_embedding":
            model = BertForPreTraining.from_pretrained( self.config.pretrained_model,  output_hidden_states=True)
            tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_model)

        return model, tokenizer

    def _load_cdsFms_model(self):
        if self.config.task == "do_finetune":
            model = AutoEnCodon.from_pretrained(self.config.pretrained_model)
            tokenizer = model.tokenizer
            model = model.model

            # Print number of trainable parameters
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            print("encodon_Classfier\nTrainable Parameter: "+ str(params))

            if self.config.finetune_type == "lora":
                # 添加LoRA适配器
                peft_config = LoraConfig(
                    r=self.config.lora_rank,
                    lora_alpha=1,
                    target_modules=["query", "key", "value", "dense"],
                    bias="lora_only"
                )
                model = inject_adapter_in_model(peft_config, model)
            else:
                # """Freezes all parameters in the model"""
                print("encodon_Classfier\nTrainable Parameter:") 
                for (param_name, param) in model.named_parameters(): 
                    param.requires_grad = False 
            
            # Unfreeze the prediction head
            for (param_name, param) in model.classifier.named_parameters():
                param.requires_grad = True  

            # Print trainable Parameter          
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            print("encodon_Classfier\nTrainable Parameter: "+ str(params) + "\n")

        elif self.config.task == "do_embedding":
            model = AutoEnCodon.from_pretrained(self.config.pretrained_model)
            tokenizer = model.tokenizer
            model = model.model

        return model, tokenizer

    def _load_mcodon_model(self):
        if self.config.task == "do_finetune":
            tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_model, trust_remote_code=True)
            model = AutoModelForSequenceClassification.from_pretrained(self.config.pretrained_model, trust_remote_code=True) 

            class_config = MCCodonClassConfig(num_labels=self.config.num_labels, model_name = self.config.pretrained_model)
            model = MCCodonForSimpleSequenceClassification(model.config, class_config)

            # tokenizer.eos_token = tokenizer.pad_token
            tokenizer.eos_token='[EOS]'
            tokenizer.pad_token = '[PAD]'
            model.config.pad_token_id = tokenizer.pad_token_id

            # Print number of trainable parameters
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            print("mcodon\nTrainable Parameter: "+ str(params))

            if self.config.finetune_type == "lora":
                # 添加LoRA适配器
                peft_config = LoraConfig(
                    r=self.config.lora_rank,
                    lora_alpha=1,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    bias="lora_only"
                )
                model = inject_adapter_in_model(peft_config, model)
            else:
                # """Freezes all parameters in the model"""
                print("mcodon_freeze_Classfier\nTrainable Parameter:") 
                for (param_name, param) in model.named_parameters(): 
                    param.requires_grad = False

            # Unfreeze the prediction head
            for (param_name, param) in model.classifier.named_parameters():
                param.requires_grad = True  

            # Print trainable Parameter          
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            print("mcodon_Classfier\nTrainable Parameter: "+ str(params) + "\n")
        
        elif self.config.task == "do_embedding":
            tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_model, trust_remote_code=True) 
            model = AutoModel.from_pretrained(self.config.pretrained_model, trust_remote_code=True)

        return model, tokenizer

    def _load_cdsbert_model(self):
        if self.config.task == "do_finetune":
            """加载BERT系列模型"""
            config = BertConfig.from_pretrained(self.config.pretrained_model)
            tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_model)
            class_config = BERTClassConfig(num_labels=self.config.num_labels)
            model = BertForSimpleSequenceClassification(config, class_config)

            # Print number of trainable parameters
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            print("cdsbert_Classfier\nTrainable Parameter: "+ str(params))

            if self.config.finetune_type == "lora":

                # 添加LoRA适配器
                peft_config = LoraConfig(
                    r=self.config.lora_rank,
                    lora_alpha=1,
                    target_modules=["query", "key", "value", "dense"],
                    bias="lora_only"
                )
                model = inject_adapter_in_model(peft_config, model)
            else:
                print("cdsbert_freeze_Classfier\nTrainable Parameter:")
                # """Freezes all parameters in the model"""
                for (param_name, param) in model.named_parameters(): 
                    param.requires_grad = False

            # Unfreeze the prediction head
            for (param_name, param) in model.classifier.named_parameters():
                param.requires_grad = True

            # Print trainable Parameter          
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            print("cdsbert_LoRA_Classfier\nTrainable Parameter: "+ str(params) + "\n")
        
        elif self.config.task == "do_embedding":
            
            model = EsmModel.from_pretrained(self.config.pretrained_model)
            tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_model)

        return model, tokenizer

    def _load_esm_model(self):

        if self.config.task == "do_finetune":
            """加载ESM系列模型"""
            model = AutoModelForSequenceClassification.from_pretrained(
                self.config.pretrained_model,
                num_labels=self.config.num_labels
            )
            tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_model)

            # Print number of trainable parameters
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            print("ESM_Classfier\nTrainable Parameter: "+ str(params))

            if self.config.finetune_type == "lora":
                # 添加LoRA适配器
                peft_config = LoraConfig(
                    r=self.config.lora_rank,
                    lora_alpha=1,
                    target_modules=["query", "key", "value", "dense"],
                    bias="lora_only"
                )
                model = inject_adapter_in_model(peft_config, model)
            elif self.config.finetune_type == "full":
                for (param_name, param) in model.named_parameters(): 
                    param.requires_grad = True
            elif self.config.finetune_type == "deep":
                pass
            
            elif self.config.finetune_type == "shallow":
                pass

            else:
                print("ESM_freeze_Classfier\nTrainable Parameter:")
                # """Freezes all parameters in the model"""
                for (param_name, param) in model.named_parameters(): 
                    param.requires_grad = False

            # Unfreeze the prediction head
            for (param_name, param) in model.classifier.named_parameters():
                param.requires_grad = True

            # Print trainable Parameter          
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            print("ESM_LoRA_Classfier\nTrainable Parameter: "+ str(params) + "\n")
        
        elif self.config.task == "do_embedding":
            
            model = EsmModel.from_pretrained(self.config.pretrained_model)
            tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_model)

        return model, tokenizer

    def _load_t5_model(self):

        if self.config.task == "do_finetune":
            """加载T5系列模型"""
            model = T5EncoderModel.from_pretrained(self.config.pretrained_model)
            if "ankh" in self.config.pretrained_model.lower():
                tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_model)
            else :
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
            
            elif self.config.finetune_type == "full":
    
                for (param_name, param) in model.named_parameters(): 
                    param.requires_grad = True
                print("T5_full_Classfier\nTrainable Parameter:") 

            elif self.config.finetune_type == "deep":

                for (param_name, param) in model.named_parameters(): 
                    param.requires_grad = False
                for (param_name, param) in model.encoder.block[22].named_parameters(): 
                    param.requires_grad = True
                for (param_name, param) in model.encoder.block[23].named_parameters(): 
                    param.requires_grad = True
                print("T5_deep_Classfier\nTrainable Parameter:") 
                
            elif self.config.finetune_type == "shallow":
                
                for (param_name, param) in model.named_parameters(): 
                    param.requires_grad = False
                for (param_name, param) in model.encoder.block[0].named_parameters(): 
                    param.requires_grad = True
                for (param_name, param) in model.encoder.block[1].named_parameters(): 
                    param.requires_grad = True
                print("T5_shallow_Classfier\nTrainable Parameter:")

            else:
                # """Freezes all parameters in the model"""
                print("T5_freeze_Classfier\nTrainable Parameter:") 
                for (param_name, param) in model.named_parameters(): 
                    param.requires_grad = False

            # Unfreeze the prediction head
            for (param_name, param) in model.classifier.named_parameters():
                param.requires_grad = True
            # Print trainable Parameter    
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            print("T5_Classfier\nTrainable Parameter: "+ str(params) + "\n") 

        elif self.config.task == "do_embedding":
            model = T5EncoderModel.from_pretrained(self.config.pretrained_model)
            tokenizer = T5Tokenizer.from_pretrained(self.config.pretrained_model)

        return model, tokenizer

    def _load_protbert_model(self):
    
        if self.config.task == "do_finetune":
            """加载protbert系列模型"""
            config = BertConfig.from_pretrained(self.config.pretrained_model)
            tokenizer = BertTokenizer.from_pretrained(self.config.pretrained_model, do_lower_case=False)

            # Create new Classifier model with BERT dimensions
            class_config = BERTClassConfig(num_labels=self.config.num_labels)
            model = BertForSimpleSequenceClassification(config, class_config)
            
            # Print number of trainable parameters
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            print("protbert_Classfier\nTrainable Parameter: "+ str(params))

            if self.config.finetune_type == "lora":

                # 添加LoRA适配器
                peft_config = LoraConfig(
                    r=self.config.lora_rank,
                    lora_alpha=1,
                    target_modules=["query", "key", "value", "dense"],
                    bias="lora_only"
                )
                model = inject_adapter_in_model(peft_config, model)
            else:
                print("protbert_freeze_Classfier\nTrainable Parameter:")
                # """Freezes all parameters in the model"""
                for (param_name, param) in model.named_parameters(): 
                    param.requires_grad = False

            # Unfreeze the prediction head
            for (param_name, param) in model.classifier.named_parameters():
                param.requires_grad = True

            # Print trainable Parameter          
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            print("protbert_LoRA_Classfier\nTrainable Parameter: "+ str(params) + "\n")
        
        elif self.config.task == "do_embedding":
            tokenizer = BertTokenizer.from_pretrained(self.config.pretrained_model, do_lower_case=False)
            model = BertModel.from_pretrained(self.config.pretrained_model)

        return model, tokenizer


class DataProcessor:
    """数据预处理处理器"""
    def __init__(self, tokenizer, config: ModelConfig):

        self.tokenizer = tokenizer
        self.config = config
        self.codon_to_aa = self._init_cdsbertdata()

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
            return self._codon_tokenize(sequence)
            
        # T5系列的特殊处理
        if "t5" in self.config.pretrained_model.lower() or "prot_bert" in self.config.pretrained_model.lower() or "ankh" in self.config.pretrained_model.lower():
            sequence = re.sub(r'[OBUZJ]', 'X', sequence)
            sequence = " ".join(list(sequence))
            if "prostt5" in self.config.pretrained_model.lower():
                    sequence = f"<AA2fold> {sequence}"
        
        # bioobang的特殊处理
        if "bioobang" in self.config.pretrained_model.lower():
            sequence = sequence.upper()
        
        if "codonbert" in self.config.pretrained_model.lower():
            sequence = self._codon_tokenize(sequence)
        
        if "esm" in self.config.pretrained_model.lower():
           sequence = re.sub(r'[OBUZJ]', 'X', sequence) 

        if "encodon" in self.config.pretrained_model.lower():
            sequence = f"<CLS>{sequence}<SEP>"  #<CLS>AAAAAA<SEP>

        if "mistral-codon" in self.config.pretrained_model.lower():
            sequence = self._codon_tokenize(sequence)

        if "cdsbert" in self.config.pretrained_model.lower():
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

        if "t5" in self.config.pretrained_model.lower() or "prot_bert" in self.config.pretrained_model.lower() or "ankh" in self.config.pretrained_model.lower() or "esm" in self.config.pretrained_model.lower():
            tokenized = self.tokenizer(processed_seqs,
                                        max_length=self.config.max_seq_length, # 1024
                                        padding=True,
                                        truncation=True)       
            return Dataset.from_dict({**tokenized, "sequence":df["sequence"], "labels": df['label']})
        
        elif "calm" in self.config.pretrained_model.lower():
            tokenized = self.tokenizer(
                    processed_seqs, 
                    max_length=self.config.max_seq_length,
                    padding=True,          # 填充到相同长度
                    truncation=True,        # 截断到最大长度
                )
            return Dataset.from_dict({**tokenized, "sequence":df["sequence"], "labels": df['label']})

        elif "bioobang" in self.config.pretrained_model.lower():
            tokenized = self.tokenizer(processed_seqs,
                                        max_length=self.config.max_seq_length,    #1024
                                        padding=True,
                                        truncation=True)
            # tokenized = self.tokenizer(processed_seqs)
            return Dataset.from_dict({**tokenized, "sequence":df["sequence"], "labels": df['label']}) 
        
        elif "codonbert" in self.config.pretrained_model.lower():

            # self.config.max_seq_length = max([len(sublist.split(" ")) for sublist in processed_seqs])
            tokenized = self.tokenizer(processed_seqs,
                                        max_length=self.config.max_seq_length,    #1024
                                        padding='max_length',
                                        truncation=True,
                                        return_special_tokens_mask=True)
            return Dataset.from_dict({**tokenized, "labels": df['label']})

        elif "encodon" in self.config.pretrained_model.lower():

            tokenized = self.tokenizer(processed_seqs,
                                        max_length=self.config.max_seq_length,    #2048
                                        padding='max_length',
                                        add_special_tokens=False,
                                        truncation=True,
                                        return_special_tokens_mask=True)
            return Dataset.from_dict({**tokenized, "labels": df['label']})

        elif "mistral-codon" in self.config.pretrained_model.lower():

            tokenized = self.tokenizer(
                processed_seqs,
                max_length = self.config.max_seq_length,  #
                truncation=True,
                padding=True,
            )
            return Dataset.from_dict({**tokenized, "labels": df['label']})

        elif "cdsbert" in self.config.pretrained_model.lower():
            tokenized = self.tokenizer(processed_seqs,
                                        max_length=self.config.max_seq_length,   ##  
                                        padding=True,
                                        truncation=True)       
            return Dataset.from_dict({**tokenized, "labels": df['label']})

    def _translate_rna_to_single_letter_codon(self, rna_sequence: str) -> str:
        # 按照每3个碱基分割成密码子
        codons = [rna_sequence[i:i+3] for i in range(0, len(rna_sequence), 3)]
        
        # 查找对应的氨基酸
        aa_sequence = [self.codon_to_aa.get(codon, '?') for codon in codons]
        
        return ''.join(aa_sequence)

    def _init_cdsbertdata(self) -> dict:
        # 数据：Codon与Single Letter Codon的对应关系
        data = [
            ("GCU", "a"), ("GCC", "A"), ("GCA", "@"), ("GCG", "b"),
            ("CGU", "B"), ("CGC", "#"), ("CGA", "$"), ("CGG", "%"),
            ("AGA", "r"), ("AGG", "R"), ("AAU", "n"), ("AAC", "N"),
            ("GAU", "d"), ("GAC", "D"), ("UGU", "c"), ("UGC", "C"),
            ("GAA", "e"), ("GAG", "E"), ("CAA", "q"), ("CAG", "Q"),
            ("GGU", "^"), ("GGC", "G"), ("GGA", "&"), ("GGG", "g"),
            ("CAU", "h"), ("CAC", "H"), ("AUU", "i"), ("AUC", "I"),
            ("AUA", "j"), ("UUA", "+"), ("UUG", "M"), ("CUU", "m"),
            ("CUA", "J"), ("CUG", "L"), ("UAU", "y"), ("UAC", "Y"),
            ("GUU", "u"), ("GUC", "v"), ("GUA", "U"), ("GUG", "V"),
            ("UAA", "]"), ("UAG", "}"), ("UGA", ")")
        ]

        # 创建 DataFrame
        df = pd.DataFrame(data, columns=["Codon", "Single Letter Codon"])

        # 构建密码子到单字母氨基酸的映射字典
        codon_to_aa = dict(zip(df['Codon'], df['Single Letter Codon']))

        return codon_to_aa
