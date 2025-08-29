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

# 配置日志记录 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__) 

class ModelEvaluator:
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
            pretrained_model_type = self.config["pretrained_model_type"],
            num_labels=self.config['num_labels'],
            train=self.config["train"]

        )


    def _load_finetune_model(self) -> None:
        """Load pretrained model with adapters."""
        try:
            # 初始化模型和分词器，根据模型名称选择加载方式
            non_frozen_params = torch.load( os.path.join( self.config["finetuned_params_dir"], self.config["finetuned_params_name"]  ) )

            # # # # Assign the non-frozen parameters to the corresponding parameters of the model
            for param_name, param in self.model.named_parameters():
                if param_name in non_frozen_params:
                    param.data = non_frozen_params[param_name].data
            
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

    def _load_fundation_model(self) -> None:
        """Load pretrained model with adapters."""
        try:
            # 初始化模型和分词器，根据模型名称选择加载方式
            model_config = self._get_model_config()
            classifier = SequenceClassifier(model_config)
           
            # 准备tokenizer和数据处理器
            if "calm-t5" in model_config.pretrained_model.lower():
                calm_tokenizer, t5_tokenizer = classifier.calm_tokenizer, classifier.t5_tokenizer
                calm_t5_tokenizer = calm_tokenizer, t5_tokenizer
                self.model, self.tokenizer = classifier.model, calm_t5_tokenizer
                # processor = DataProcessor(calm_t5_tokenizer, model_config)
            else:
                # processor = DataProcessor(classifier.tokenizer, model_config)
                self.model, self.tokenizer = classifier.model, classifier.tokenizer
        
            # self.model.load_state_dict(state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def _do_evaluate(self) -> Dict[str, float]:

        true_labels, pred_probs = [], []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="do inference"):
                outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                    )

                pred_probs.extend(torch.softmax(outputs.logits, dim=1).cpu().numpy())
                true_labels.extend(batch['labels'].cpu().numpy())

        return self._calculate_metrics(true_labels, pred_probs)
    
    def _calculate_metrics(
            self,
            true_labels: List[int],
            pred_probs: List[List[float]],
        ) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        num_classes = self.config["num_labels"]
        pred_labels = np.argmax(pred_probs, axis=1)
        true_labels = np.array(true_labels)
        pred_probs = np.array(pred_probs)
        metrics_dict = {}

        if num_classes == 2:
            metrics_dict.update({
                "AUC": metrics.roc_auc_score(np.eye(num_classes)[true_labels], pred_probs),
                "ACC": metrics.accuracy_score(true_labels, pred_labels),
                "Precision": metrics.precision_score(true_labels, pred_labels),
                "Recall": metrics.recall_score(true_labels, pred_labels),
                "F1": metrics.f1_score(true_labels, pred_labels),
                "MCC": metrics.matthews_corrcoef(true_labels, pred_labels),
                "sensitivity": metrics.recall_score(true_labels, pred_labels),
                "specificity": self._specificity(true_labels, pred_labels)
            })
        else:
            metrics_dict.update({
                "AUC": metrics.roc_auc_score( np.eye(num_classes)[true_labels], pred_probs, average="macro" ),
                "ACC": metrics.accuracy_score(true_labels, pred_labels),
                "Precision": metrics.precision_score(true_labels, pred_labels, average="macro"),
                "Recall": metrics.recall_score( true_labels, pred_labels, average="macro" ),
                "F1": metrics.f1_score(true_labels, pred_labels, average="macro"),
                "MCC": metrics.matthews_corrcoef(true_labels, pred_labels)
            })

        logger.info("\n%s", pd.DataFrame([metrics_dict]))
        return metrics_dict
    
    @staticmethod
    def _specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity for binary classification."""
        tn, fp, _, _ = metrics.confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp)
    
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
            self._load_fundation_model()

        metrics = self._do_evaluate()
        
        output_dir = os.path.join(self.config['finetuned_params_dir'], "evaluation_results")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "metrics.csv")
        pd.DataFrame([metrics]).to_csv(output_path)
        logger.info(f"Evaluation results saved to {output_path}")

def main(evaluator_config):
    try:
        evaluator = ModelEvaluator(evaluator_config)
        evaluator.run()
    except Exception as e:   
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":  

    parser = argparse.ArgumentParser(description='ProteinBLM Training and Evaluation') 
    parser.add_argument('--pretrained_model', '-p', default="/hpcfs/fhome/yangchh/workdir/self/TLCP-EPE/FMs/pretrain_codon/huggingface_calm", help='预训练模型名称或路径')
    parser.add_argument('--pretrained_model_type', '-pmt', type=str, default="codon", help='dna、rna、codon、protein')

    parser.add_argument('--finetune_path', '-fp', default="/hpcfs/fhome/yangchh/workdir/self/TLCP-EPE/method/CaLM/mpepe/output/saved_models_huggingface_calm_lora_fold_10", help='微调模型参数路径')
    parser.add_argument('--finetuned_params_name', '-fpn', type=str, default="fold_10_finetuned_params.pth", help='微调参数文件的名字')  

    parser.add_argument('--data_path', '-dp', default="/hpcfs/fhome/yangchh/workdir/self/TLCP-EPE/test/test_data", help='训练数据路径') #/hpcfs/fhome/yangchh/workdir/self/TLCP-EPE/test/nesg #/hpcfs/fhome/yangchh/workdir/self/TLCP-EPE/test/ecoli_ATCC
    parser.add_argument('--file_name', '-fn', type=str, default="ecoli_test.csv", help='文件的名字')  #nesg_high_low_<0.75.csv   ecoli_ATCC<0.75.csv

    parser.add_argument('--num_labels', '-n', type=int, default=2, help='标签的个数')
    parser.add_argument('--batch_size', '-b', type=int, default=2, help='训练批次大小') 
    parser.add_argument('--max_seq_length', '-m', type=int, default=1024, help='最大长度')
    parser.add_argument('--task', '-t', type=str, default="do_finetune", help='do_finetune, do_inference, do_embedding')
    args = parser.parse_args()

    # finetuned_params_dir = os.path.join( args.finetune_path, "saved_models_"+args.pretrained_model.split("/")[-1]  + "_loraCaLMMLP" ) 
    finetuned_params_dir = args.finetune_path

    evaluator_config = {
            'pretrained_model': args.pretrained_model,
            'pretrained_model_type': args.pretrained_model_type,
            'data_path': args.data_path,
            "file_name":args.file_name,
            'finetuned_params_dir': finetuned_params_dir,
            'finetuned_params_name':args.finetuned_params_name,
            'num_labels': args.num_labels,
            'batch_size': args.batch_size, 
            'max_seq_length': args.max_seq_length,
            'task': args.task,
            'train': False
        }
    main(evaluator_config)