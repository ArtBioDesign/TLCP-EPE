import os
import logging
import argparse
from dataclasses import dataclass
from typing import Dict, List, Any
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from evaluate import load
from T5E import *
from Bio.Seq import Seq
import os,sys  
from sklearn import metrics
from torch.utils.data import DataLoader
from utils import *
from tqdm import tqdm

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

# 配置日志记录 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__) 

class EmbeddingExtractor:
    """模型评估处理器"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_config = self._get_model_config()

        classifier = SequenceClassifier(model_config)
        self.model, self.tokenizer = classifier.model, classifier.tokenizer
        self.processor = DataProcessor(self.tokenizer, model_config)

        self.data_df = None
        self.dataset = None
        self.dataloader = None

    def _get_model_config(self) -> ModelConfig:
        """生成 ModelConfig 对象，避免代码重复"""
        return ModelConfig(
            pretrained_model = self.config['pretrained_model'],
            max_seq_length = self.config['max_seq_length'],
            task = self.config['task'],
            pretrained_model_type = self.config["pretrained_model_type"]
        )
    
    def _load_finetune_model(self) -> None:
        """Load pretrained model with adapters."""
        try:
            # 初始化模型和分词器，根据模型名称选择加载方式
            non_frozen_params = torch.load(self.config["finetuned_params_path"])

            # # # # Assign the non-frozen parameters to the corresponding parameters of the model
            for param_name, param in self.model.named_parameters():
                if param_name in non_frozen_params:
                    param.data = non_frozen_params[param_name].data
            
            # from safetensors.torch import load_file
            # weights_path = "/hpcfs/fhome/yangchh/workdir/self/TLCP-EPE/data/output/saved_models_prot_t5_xl_uniref50_lora_fold_8/results/checkpoint-768/model.safetensors"
            # state_dict = load_file(weights_path)
            # self.model.load_state_dict(state_dict)  # strict=False 允许部分加载

            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _load_foundation_model(self)-> None:
        """Load pretrained model with adapters."""
        try: 
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def _do_embedding(self) -> Dict[str, float]:
        
        embedding_repr_list = []
        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="Extracting embeddings"):
                
                if "calm" in self.config['pretrained_model'].lower():
                    result = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        repr_layers=[12]
                        )
                    if self.config["task"] == "do_finetune":
                        repr_ = result["hidden_states"]
                    else:
                        repr_ = result["representations"][12]
                    embedding_repr = repr_.mean(axis=1).cpu().numpy() 
                else:
                    result = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        output_hidden_states=True
                    )
                    if self.config["task"] == "do_finetune":
                        embedding_repr = result["hidden_states"][-1].detach().mean(1).cpu().numpy()
                    else:
                        embedding_repr = result.last_hidden_state.detach().mean(1).cpu().numpy()

                for emb in embedding_repr:
                    embedding_repr_list.append(emb.tolist())
        embedding_dataset = self.dataset.add_column('embedding', embedding_repr_list)
        df = embedding_dataset.to_pandas()

        output_dir = os.path.join(self.config['data_path'], f"embeddings")
        os.makedirs(output_dir, exist_ok=True)

        model_name = self.config['pretrained_model'].split("/")[-1]
        output_path = os.path.join(output_dir, f"{model_name}.pkl")

        df.to_pickle(output_path)

        logger.info(f"Predict results saved to {output_path}")

    def run(self) -> None:
        
        self.data_df = self.processor._load_data( os.path.join(self.config["data_path"], self.config["file_name"] ) )

        self.dataset = self.processor._create_dataset(self.data_df)
        
        self.dataloader = DataLoader( self.dataset.with_format("torch", device=self.device),
                                 batch_size=self.config['batch_size'],
                                 shuffle=False,
                                 num_workers=0) 

        if self.config["task"] == "do_finetune":
            self._load_finetune_model()
        elif self.config["task"] == "do_embedding":
            self._load_foundation_model()
        
        self._do_embedding()
        
def main(evaluator_config):

    try:
        evaluator = EmbeddingExtractor(evaluator_config)
        evaluator.run()
    except Exception as e:   
        logger.error(f"Embedding failed: {str(e)}")
        raise

if __name__ == "__main__":  

    parser = argparse.ArgumentParser(description='Foundation Models embeddings')  
    # parser.add_argument('--pretrained_model', '-p', default="/hpcfs/fhome/yangchh/workdir/self/TLCP-EPE/FMs/pretrain_codon/huggingface_calm", help='Fms模型名称或路径')
    # parser.add_argument('--pretrained_model_type', '-pmt', type=str, default="codon", help='dna、rna、codon、protein')
    # parser.add_argument('--finetuned_params_path', '-fpn', type=str, default="/hpcfs/fhome/yangchh/workdir/self/TLCP-EPE/data/output/saved_models_huggingface_calm_lora_fold_4/fold_4_finetuned_params.pth", help='微调参数文件的名字') #

    parser.add_argument('--pretrained_model', '-p', default="Rostlab/prot_t5_xl_uniref50", help='Fms模型名称或路径')
    parser.add_argument('--pretrained_model_type', '-pmt', type=str, default="protein", help='dna、rna、codon、protein')
    parser.add_argument('--finetuned_params_path', '-fpn', type=str, default="/hpcfs/fhome/yangchh/workdir/self/TLCP-EPE/data/output/saved_models_prot_t5_xl_uniref50_lora_fold_4/fold_4_finetuned_params.pth", help='微调参数文件的名字') #
    parser.add_argument('--data_path', '-dp', default="/hpcfs/fhome/yangchh/workdir/self/TLCP-EPE/test/Aldolase_expression_test", help='原始数据路径')
    parser.add_argument('--file_name', '-fn', type=str, default="aldolase_expression.csv", help='文件的名字')
    parser.add_argument('--max_seq_length', '-m', type=int, default=None, help='序列的最大长度')
    parser.add_argument('--batch_size', '-b', type=int, default=2, help='批量大小')
    parser.add_argument('--task', '-t', type=str, default="do_finetune", help='do_finetune, do_inference, do_embedding')
    args = parser.parse_args() 

    embedding_config = {
            'pretrained_model': args.pretrained_model,
            'pretrained_model_type': args.pretrained_model_type,
            'finetuned_params_path': args.finetuned_params_path,
            'data_path': args.data_path,
            'batch_size': args.batch_size, 
            'max_seq_length': args.max_seq_length,
            'file_name': args.file_name,
            'task': args.task  
        }
    main(embedding_config)