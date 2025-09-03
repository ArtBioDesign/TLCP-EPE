from dataclasses import dataclass
import torch

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

T5_PLM_CONFIG = {
            'pretrained_model': "Rostlab/prot_t5_xl_uniref50",
            'pretrained_model_type': "protein",
            'finetuned_params_path': "saved_models_prot_t5_xl_uniref50_lora_fold_4/fold_4_finetuned_params.pth",
            'data_path': "/hpcfs/fhome/yangchh/workdir/self/TLCP-EPE/data",
            'batch_size': 4, 
            'max_seq_length': None,
            'file_name': "sample.csv",
            'task': "do_finetune",
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
        }

CALM_PLM_CONFIG = {
            'pretrained_model': "/hpcfs/fhome/yangchh/workdir/self/TLCP-EPE/FMs/pretrain_codon/huggingface_calm",
            'pretrained_model_type': "codon",
            'finetuned_params_path': "saved_models_huggingface_calm_lora_fold_4/fold_4_finetuned_params.pth",
            'data_path': "/hpcfs/fhome/yangchh/workdir/self/TLCP-EPE/data",
            'batch_size': 16, 
            'max_seq_length': None,
            'file_name': 'sample.csv',
            'task': 'do_finetune',
            'device' : torch.device('cuda' if torch.cuda.is_available() else 'cpu')     
        }