from dataclasses import dataclass
import torch
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

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
  
@dataclass
class PLM_Config:

    T5_PLM_CONFIG = {
                'pretrained_model': "",
                'pretrained_model_type': "protein",
                'finetuned_params_path': "",
                'data_path': "",
                'batch_size': 4, 
                'max_seq_length': None,
                'file_name': "sample.csv",
                'task': "do_finetune",
                'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
            }

    CALM_PLM_CONFIG = {
                'pretrained_model': "",
                'pretrained_model_type': "codon",
                'finetuned_params_path': "",
                'data_path': "",
                'batch_size': 16, 
                'max_seq_length': None,
                'file_name': 'sample.csv',
                'task': 'do_finetune',
                'device' : torch.device('cuda' if torch.cuda.is_available() else 'cpu')     
            }