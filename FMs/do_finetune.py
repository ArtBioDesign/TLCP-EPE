import os
import logging
import argparse
import wandb
import torch
from sklearn import metrics
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import (
    TrainingArguments,
    Trainer,
    set_seed
)
from evaluate import load
from utils import *
import os,sys,re
import wandb
wandb.init(project="your_project", mode="offline")

# 配置日志记录 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
) 
logger = logging.getLogger(__name__)

class TrainerWrapper:
    def __init__(self, model, tokenizer, config: ModelConfig):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer

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


        # return metric.compute(predictions=predictions, references=eval_pred.label_ids)
    
    def save_finetuned_parameters(self, filepath):
        non_frozen_params = {}
        for param_name, param in self.model.named_parameters():
            if param.requires_grad:  # 保存所有参与训练的参数
                non_frozen_params[param_name] = param
        torch.save(non_frozen_params, filepath)
        logger.info(f"Finetuned parameters saved to {filepath}")
    
    def train(self, train_data: Dataset, valid_data: Dataset) -> Trainer:

        results = self.config.save_model_path

        training_args = TrainingArguments(
            output_dir=results,
            evaluation_strategy="epoch",
            logging_strategy = "epoch",
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
            seed=self.config.seed,
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

        if "calm" in config.pretrained_model.lower():
            logits = logits[0]

        probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()

        y_pred = np.argmax(logits, axis=1)

        results_df = pd.DataFrame({
            'fold': 0,
            'y_true': y_true,
            'y_pred': y_pred,
            'prob_class_0': probabilities[:, 0],
            'prob_class_1': probabilities[:, 1]
        })

        results_file = os.path.join(results, f'fold_{0}_predictions.csv')
        results_df.to_csv(results_file, index=False)
        logger.info(f"Fold {0} predictions saved to {results_file}")

        return trainer

def main(config: ModelConfig):
    """主执行流程"""
    try:
        # 设置随机种子  
        set_seed(config.seed)

        # 初始化组件
        classifier = SequenceClassifier(config)
        processor = DataProcessor(classifier.tokenizer, config)
        
        # 加载数据
        logger.info("Loading data...")
        df = processor._load_data( os.path.join(config.data_path, config.file_name ) )
        # 数据划分
        train_df, valid_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

        # 创建数据集
        train_data = processor._create_dataset(train_df)
        valid_data = processor._create_dataset(valid_df)

        # 创建保存模型及微调参数的路径
        name = config.pretrained_model.split("/")[-1] 
        output_dir = os.path.join(config.data_path, f"checkpoints/saved_models_{ name }")
        os.makedirs(output_dir, exist_ok=True)
        config.save_model_path = output_dir
        
        # 训练模型
        logger.info("Starting training...")
        trainer_wrapper = TrainerWrapper(classifier.model, classifier.tokenizer, config)
        trainer = trainer_wrapper.train(train_data, valid_data)

        # 保存模型及微调参数的路径
        trainer_wrapper.save_finetuned_parameters( os.path.join(config.save_model_path, config.finetuned_params_name) )
        
        trainer.save_model(config.save_model_path) 
        logger.info(f"Model saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ProteinBLM Training')  
    parser.add_argument('--pretrained', '-p', default="lonelycrab88/BiooBang-1.0", help='预训练模型名称或路径')
    parser.add_argument('--pretrained_model_type', '-pmt', type=str, default="codon", help='dna、rna、codon、protein')
    parser.add_argument('--data_path', '-dp', default="/hpcfs/fhome/yangchh/workdir/self/TLCP-EPE/test/mpepe", help='训练数据路径')
    parser.add_argument('--file_name', '-fn', type=str, default="mpepe_bc.csv", help='文件的名字')
    parser.add_argument('--finetuned_params_name', '-fpn', type=str, default="finetuned_params.pth", help='微调参数文件的名字')  
    parser.add_argument('--epochs', '-e', type=int, default=30, help='训练轮数')
    parser.add_argument('--batch_size', '-b', type=int, default=2, help='训练批次大小')
    parser.add_argument('--lr', '-l', type=float, default=0.00005, help='学习率')  
    parser.add_argument('--num_labels', '-n', type=int, default=2, help='标签的个数')
    parser.add_argument('--task', '-t', type=str, default="do_finetune", help='do_finetune, do_inference, do_embedding')
    parser.add_argument('--max_seq_length', '-m', type=int, default=None, help='最大长度')
    parser.add_argument('--finetune_type', '-ft', type=str, default="full", help='lora, freeze, full, shallow, deep')
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
        finetune_type = args.finetune_type,
    )

    main(config) 
