import os
import logging
import argparse
import torch
import numpy as np
import pandas as pd
from transformers import (
    TrainingArguments,
    Trainer,
    set_seed
)
from evaluate import load
from sklearn.model_selection import StratifiedKFold
import re
from utils import *
import os,sys
from typing import List, Dict, Union, Any, Optional
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from sklearn import metrics
import json
import random
import wandb
wandb.init(project="your_project", mode="offline")

# 配置日志记录 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
) 

logger = logging.getLogger(__name__)

def set_global_random_seed(seed):
    """设置所有相关的随机种子"""
    os.environ['PYTHONHASHSEED'] = str(seed)  # Python hash种子
    np.random.seed(seed)  # Numpy随机种子
    torch.manual_seed(seed)  # PyTorch CPU随机种子
    torch.cuda.manual_seed(seed)  # PyTorch GPU随机种子
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False  # 禁用cudnn自动寻找最快的卷积算法
    random.seed(seed)  # Python随机种子

class TrainerWrapper:
    """训练流程包装器"""
    def __init__(self, model, tokenizer, config: ModelConfig):
        
        # 确保模型初始化的确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.data_collator = None

    def compute_metrics(self, eval_pred):

        """计算多个评估指标：AUC, Accuracy, Precision, Recall, F1"""
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids

        if self.config.num_labels > 1:
            # 对于分类任务，需要处理预测结果
            try:
                # 获取预测概率
                probs = torch.nn.functional.softmax(torch.tensor(predictions), dim=1).numpy()
                # 获取预测类别
                pred_labels = np.argmax(predictions, axis=1)
            except:
                # 如果predictions已经是元组，取第一个元素（这种情况可能发生在某些模型中）
                probs = torch.nn.functional.softmax(torch.tensor(predictions[0]), dim=1).numpy()
                pred_labels = np.argmax(predictions[0], axis=1)
            # 计算各项指标
            evo_metrics = {}
            # 计算 AUC（多分类使用 macro 平均）
            try:
                if self.config.num_labels == 2:
                    # metrics['eval_auc'] =   #roc_auc_score(labels, probs[:, 1]) 
                    evo_metrics['eval_auc'] = metrics.roc_auc_score(labels, probs[:, 1])  # 使用正类的概率
                else:
                    evo_metrics['eval_auc'] = metrics.roc_auc_score(labels, probs, multi_class='ovr', average='macro')
            except Exception as e:
                logger.warning(f"AUC calculation failed: {str(e)}")
                metrics['eval_auc'] = float('nan')
            # 计算其他指标
            
            evo_metrics['eval_accuracy'] = metrics.accuracy_score(labels, pred_labels)
            evo_metrics['eval_precision'] = metrics.precision_score(labels, pred_labels)
            evo_metrics['eval_recall'] = metrics.recall_score(labels, pred_labels)
            evo_metrics['eval_f1'] = metrics.f1_score(labels, pred_labels)

            return evo_metrics
        else:
            # 对于回归任务保持原有的评估方式
            metric = load("spearmanr")
            return metric.compute(predictions=predictions, references=labels)

    def save_finetuned_parameters(self, filepath):
        """保存仅微调的参数"""
        non_frozen_params = {}
        for param_name, param in self.model.named_parameters():
            if param.requires_grad:
                non_frozen_params[param_name] = param
        torch.save(non_frozen_params, filepath)
        logger.info(f"Finetuned parameters saved to {filepath}")
    
    def train_fold(self, train_data: Dataset, valid_data: Dataset, fold_num: int, output_dir: str) -> Trainer:
        """执行单折训练"""
        results = os.path.join(output_dir, "results")

        training_args = TrainingArguments(
            seed=self.config.seed,
            output_dir=results,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=5,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation,
            num_train_epochs=self.config.epochs,
            fp16=self.config.mixed_precision
        )

        trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_data,
                eval_dataset=valid_data,
                tokenizer=self.tokenizer,
                compute_metrics=self.compute_metrics
            )

        trainer.train()

        predictions = trainer.predict(valid_data)

        y_true = predictions.label_ids

        logits = predictions.predictions

        if "calm" in config.pretrained_model.lower() or "encodon" in  config.pretrained_model.lower():
            logits = logits[0]
        probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()

        y_pred = np.argmax(logits, axis=1)

        results_df = pd.DataFrame({
            'fold': [fold_num] * len(y_true),
            'y_true': y_true,
            'y_pred': y_pred,
            'prob_class_0': probabilities[:, 0],
            'prob_class_1': probabilities[:, 1]
        })

        results_file = os.path.join(output_dir, f'fold_{fold_num}_predictions.csv')
        results_df.to_csv(results_file, index=False)
        logger.info(f"Fold {fold_num} predictions saved to {results_file}")

        return trainer
  
def main(config: ModelConfig):
    """主执行流程"""
    try:
        # 设置随机种子  
        set_global_random_seed(config.seed)
        set_seed(config.seed)

        # 初始化组件
        classifier = SequenceClassifier(config)
        processor = DataProcessor(classifier.tokenizer, config)
        
        
        # 加载数据
        logger.info("Loading data...")
        # 合并训练集和验证集用于交叉验证
        all_data = processor._load_data( os.path.join(config.data_path, config.file_name ) )

        labels = all_data['label'].values

        # 初始化K折交叉验证
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=config.seed)
        fold_metrics = []

        # 执行10折交叉验证
        for fold, (train_idx, val_idx) in enumerate(skf.split(all_data, labels), 1):
            logger.info(f"Starting fold {fold}")
            
            # 准备当前折的数据
            fold_train_df = all_data.iloc[train_idx]
            fold_val_df = all_data.iloc[val_idx]
            
            # 创建数据集
            train_data = processor._create_dataset(fold_train_df)
            valid_data = processor._create_dataset(fold_val_df)
            
            # 重新初始化模型以确保每折都从相同起点开始
            classifier = SequenceClassifier(config)
            logger.info("Starting training...")
            trainer_wrapper = TrainerWrapper(classifier.model, classifier.tokenizer, config)
            
            #out_dir
            name = config.pretrained_model.split("/")[-1]
            output_dir = os.path.join(config.data_path , "output", f"saved_models_{name}_{config.finetune_type}_fold_{fold}")
            os.makedirs(output_dir, exist_ok=True)
            # 训练当前折
            trainer = trainer_wrapper.train_fold(train_data, valid_data, fold, output_dir)
            
            # 评估当前折
            eval_results = trainer.evaluate()
            logger.info(f"\nFold {fold} evaluation results:")
            for metric_name, metric_value in eval_results.items():
                logger.info(f"{metric_name}: {metric_value:.4f}")
            fold_metrics.append(eval_results)
            
            # 保存当前折的模型和参数
            finetuned_params_path = os.path.join(output_dir, f"fold_{fold}_{config.finetuned_params_name}")
            trainer_wrapper.save_finetuned_parameters(finetuned_params_path)
            
            trainer.save_model(output_dir)
            logger.info(f"Fold {fold} model saved to {output_dir}")
        
        # 计算并输出所有折的平均指标和标准差
        metrics_summary = {}
        for key in fold_metrics[0].keys():
            values = [fold[key] for fold in fold_metrics]
            metrics_summary[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        logger.info("\nCross-validation Summary:")
        for metric_name, stats in metrics_summary.items():
            logger.info(f"\n{metric_name}:")
            logger.info(f"Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}")
            logger.info(f"Min: {stats['min']:.4f}, Max: {stats['max']:.4f}")
        # 保存评估结果到文件
        results_file = os.path.join(output_dir, 'cross_validation_results.json')
        with open(results_file, 'w') as f:
            json.dump(metrics_summary, f, indent=4)
        logger.info(f"\nDetailed results saved to {results_file}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ProteinBLM Training')  
    # [Previous argument parsing code remains the same]
    parser.add_argument('--pretrained', '-p', default="lonelycrab88/BiooBang-1.0", help='预训练模型名称或路径')
    parser.add_argument('--pretrained_model_type', '-pmt', type=str, default="codon", help='dna、rna、codon、protein')
    parser.add_argument('--data_path', '-dp', default="/hpcfs/fhome/yangchh/workdir/self/TLCP-EPE/test/mpepe", help='训练数据路径')
    parser.add_argument('--file_name', '-fn', type=str, default="mpepe_bc.csv", help='文件的名字')
    parser.add_argument('--finetuned_params_name', '-fpn', type=str, default="finetuned_params.pth", help='微调参数文件的名字')  
    parser.add_argument('--epochs', '-e', type=int, default=10, help='训练轮数')       
    parser.add_argument('--batch_size', '-b', type=int, default=2, help='训练批次大小')
    parser.add_argument('--lr', '-l', type=float, default=0.0005, help='学习率')  
    parser.add_argument('--num_labels', '-n', type=int, default=2, help='标签的个数')
    parser.add_argument('--task', '-t', type=str, default="do_finetune", help='do_finetune, do_inference, do_embedding')
    parser.add_argument('--max_seq_length', '-m', type=int, default=None, help='最大长度')
    parser.add_argument('--finetune_type', '-ft', type=str, default="lora", help='lora, freeze, full, shallow, deep')
    args = parser.parse_args()

    config = ModelConfig(
        pretrained_model=args.pretrained,
        pretrained_model_type = args.pretrained_model_type, 
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_labels= args.num_labels,
        task=args.task, 
        max_seq_length=args.max_seq_length,
        finetuned_params_name=args.finetuned_params_name,
        file_name = args.file_name,
        finetune_type = args.finetune_type
    )

    main(config)

