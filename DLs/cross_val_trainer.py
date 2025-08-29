import sys
import time
import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import warnings
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    precision_score, 
    recall_score, 
    f1_score, 
    accuracy_score,
    matthews_corrcoef
)

import torch.utils.data as Data
from importlib import import_module
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim

from utils import DataProcessor, BiolmDataSet, ModelEvaluator, EarlyStopping

# 忽略警告
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

temp = ""

class CrossValidationTrainer:
    def __init__(
        self, 
        model_name='LSTM_Attention', 
        n_splits=10, 
        batch_size=128, 
        epochs=100,
        learning_rate=0.0001,
        weight_decay=1e-5,
        emdedding_model="Calm+prot_t5_xl_uniref50.pkl",
        basedir="/hpcfs/fhome/yangchh/ai/finetuneFMs/dl",
        emdedding_model_path="/hpcfs/fhome/yangchh/ai/finetuneFMs/data/embedding"
    ):
        self.set_seed(42)  # 设置全局种子
        # 数据处理
        self.data_processor = DataProcessor()
        self.emdedding_model_path = emdedding_model_path
        self.checkpoint_dir = os.path.join(basedir, f'checkpoints{temp}')
        os.makedirs(self.checkpoint_dir, exist_ok=True)  # 确保目录存在
        
        # 加载数据
        # 加载-embedding
        self.features, self.labels = self.data_processor.load_data(
            os.path.join(self.emdedding_model_path, emdedding_model)
        )

        self.X_scaled, self.y_scaled = self.data_processor.preprocess_data(
            self.features, self.labels
        )
        
        # 交叉验证配置
        self.n_splits = n_splits
        self.skf = StratifiedKFold(
            n_splits=n_splits, 
            shuffle=True, 
            random_state=42
        )
        
        # 训练配置
        self.model_name = model_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # 结果存储
        self.cv_results = []
        self.all_folds_losses = []  # 存储所有折的损失数据
    
    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _create_model(self):
        """创建模型"""
        x = import_module(f'models.{self.model_name}')
        config = x.Config()
        torch.manual_seed(42)  # 确保模型初始化一致
        model = x.Model(config).to(config.device)
        return model, config

    def _calculate_metrics(self, y_true, y_pred, y_probs):
        """
        计算详细的性能指标
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        y_probs = np.asarray(y_probs)

        unique_labels = np.unique(y_true)
        n_classes = len(unique_labels)

        metrics_dict = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'mcc': matthews_corrcoef(y_true, y_pred)
        }

        try:
            if n_classes == 2:
                metrics_dict['auc'] = roc_auc_score(y_true, y_probs[:, 1])
            else:
                metrics_dict['auc'] = roc_auc_score(
                    y_true, 
                    y_probs, 
                    multi_class='ovr', 
                    average='macro'
                )
        except Exception as e:
            print(f"AUC计算异常: {e}")
            metrics_dict['auc'] = 0.5

        return metrics_dict

    def _train_and_validate_fold(self, X_train, X_val, y_train, y_val, fold=0):
        """
        单折训练和验证
        """
        # 初始化损失记录
        fold_losses = {
            'fold': fold + 1,
            'epochs': [],
            'train_losses': [],
            'val_losses': [],
            'train_accs': [],
            'val_accs': []
        }
        
        # 创建数据集和数据加载器
        train_dataset = BiolmDataSet(X_train, y_train)
        val_dataset = BiolmDataSet(X_val, y_val)
        
        train_iter = Data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_iter = Data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # 模型初始化
        model, config = self._create_model()
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=10,
            factor=0.5,
            verbose=False
        )
        
        # 损失函数
        criterion = nn.CrossEntropyLoss().to(config.device)

        # 早停机制
        early_stopping = EarlyStopping(
            patience=30, 
            verbose=True, 
            path=os.path.join(self.checkpoint_dir, f'fold_{fold+1}_best_model.pt')
        )

        # 训练和验证指标存储
        best_val_acc = 0
        best_metrics = {}
        best_all_val_true, best_all_val_pred, best_all_val_probs = [], [], []

        for epoch in range(self.epochs):
            # 训练阶段
            model.train()
            train_loss, train_acc = 0, 0

            for trains, labels in train_iter:
                trains = trains.cuda().float()
                labels = labels.long().cuda()
                
                outputs = model(trains)
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                probabilities = F.softmax(outputs, dim=1)
                predict = torch.max(probabilities, 1)[1].cpu()
                train_acc += accuracy_score(labels.cpu(), predict)
            
            train_loss /= len(train_iter)
            train_acc /= len(train_iter)
            
            # 验证阶段
            model.eval()
            val_loss, val_acc = 0, 0
            all_val_true, all_val_pred, all_val_probs = [], [], []
            
            with torch.no_grad():
                for dev_trains, dev_labels in val_iter:
                    dev_trains = dev_trains.cuda().float()
                    dev_labels = dev_labels.long().cuda()
                    
                    dev_outputs = model(dev_trains)
                    val_loss += criterion(dev_outputs, dev_labels).item()
                    
                    dev_probabilities = F.softmax(dev_outputs, dim=1)
                    dev_predict = torch.max(dev_probabilities, 1)[1].cpu()
                    
                    all_val_true.extend(dev_labels.cpu().numpy())
                    all_val_pred.extend(dev_predict.numpy())
                    all_val_probs.extend(dev_probabilities.cpu().numpy())
                    
                    val_acc += accuracy_score(dev_labels.cpu(), dev_predict)
                
                val_loss /= len(val_iter)
                val_acc /= len(val_iter)
                
                # 记录每个epoch的损失和准确率
                fold_losses['epochs'].append(epoch + 1)
                fold_losses['train_losses'].append(train_loss)
                fold_losses['val_losses'].append(val_loss)
                fold_losses['train_accs'].append(train_acc)
                fold_losses['val_accs'].append(val_acc)
                
                # 调整学习率
                scheduler.step(val_loss)

                # 早停机制
                early_stopping(val_loss, model)
                
                # 记录val_acc最佳指标
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_metrics = self._calculate_metrics(
                        all_val_true, 
                        all_val_pred, 
                        np.array(all_val_probs)
                    )
                    best_all_val_true = all_val_true
                    best_all_val_pred = all_val_pred
                    best_all_val_probs = np.array(all_val_probs)

                # 判断是否需要提前停止
                if early_stopping.early_stop:
                    print("Early stopping triggered!")
                    break

            print(f"Fold {fold+1}, Epoch {epoch+1}: "
                  f"Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f}, "
                  f"Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}")
        
        # 保存当前折的损失数据
        self._save_fold_losses(fold_losses)
        self.all_folds_losses.append(fold_losses)
        
        self._save_fold_data(fold, best_all_val_true, best_all_val_pred, best_all_val_probs)

        # 添加折号
        best_metrics['fold'] = fold + 1
        return best_metrics

    def _save_fold_losses(self, fold_losses):
        """保存单折的损失数据到CSV文件"""
        df = pd.DataFrame({
            'epoch': fold_losses['epochs'],
            'train_loss': fold_losses['train_losses'],
            'val_loss': fold_losses['val_losses'],
            'train_acc': fold_losses['train_accs'],
            'val_acc': fold_losses['val_accs']
        })
        
        loss_file = os.path.join(
            self.checkpoint_dir, 
            f'fold_{fold_losses["fold"]}_training_history.csv'
        )
        df.to_csv(loss_file, index=False)
        print(f"已将第 {fold_losses['fold']} 折的训练历史保存到 {loss_file}")

    def save_all_folds_losses(self):
        """保存所有折的损失数据到一个合并的CSV文件"""
        all_data = []
        
        for fold_losses in self.all_folds_losses:
            fold_num = fold_losses['fold']
            for i in range(len(fold_losses['epochs'])):
                all_data.append({
                    'fold': fold_num,
                    'epoch': fold_losses['epochs'][i],
                    'train_loss': fold_losses['train_losses'][i],
                    'val_loss': fold_losses['val_losses'][i],
                    'train_acc': fold_losses['train_accs'][i],
                    'val_acc': fold_losses['val_accs'][i]
                })
        
        df = pd.DataFrame(all_data)
        combined_file = os.path.join(self.checkpoint_dir, 'all_folds_training_history.csv')
        df.to_csv(combined_file, index=False)
        print(f"已将所有折的训练历史保存到 {combined_file}")

    def _save_fold_data(self, fold, y_true, y_pred, y_probs):
        """保存每一折的真实值、预测值和预测概率到 CSV 文件""" 
        y_probs = np.array(y_probs)
        if len(y_probs.shape) == 1:
            y_probs = y_probs.reshape(-1, 1)
        df = pd.DataFrame({
            'fold': [fold + 1] * len(y_true),
            'y_true': y_true,
            'y_pred': y_pred,
        })
        for i in range(y_probs.shape[1]):
            df[f'prob_class_{i}'] = y_probs[:, i]
        fold_file = os.path.join(self.checkpoint_dir, f'fold_{fold+1}_predictions.csv')
        df.to_csv(fold_file, index=False)
        print(f"已将第 {fold+1} 折的预测结果保存到 {fold_file}")

    def run_cross_validation(self):
        """执行交叉验证"""
        cv_metrics_list = []
        
        for fold, (train_idx, val_idx) in enumerate(self.skf.split(self.X_scaled, self.y_scaled)):
            print(f"\n===== 开始第 {fold+1} 折交叉验证 =====")
            
            X_train_fold = self.X_scaled[train_idx]
            X_val_fold = self.X_scaled[val_idx]
            y_train_fold = self.y_scaled[train_idx]
            y_val_fold = self.y_scaled[val_idx]
            
            fold_metrics = self._train_and_validate_fold(
                X_train_fold, X_val_fold, 
                y_train_fold, y_val_fold, 
                fold
            )
            
            cv_metrics_list.append(fold_metrics)
        
        # 保存所有折的损失数据
        self.save_all_folds_losses()
        
        cv_summary = {
            metric: np.mean([fold[metric] for fold in cv_metrics_list if metric in fold])
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'mcc']
        }
        
        return cv_metrics_list, cv_summary

def main():
    embedding_model = "Finetune_CaLM+ProtT5.pkl" #huggingface_calm, prot_t5_xl_uniref50  Finetune_CaLM+ProtT5 Finetune_calm+esm
    basedir = "/hpcfs/fhome/yangchh/workdir/self/TLCP-EPE/DLs"
    emdedding_model_path = "/hpcfs/fhome/yangchh/workdir/self/TLCP-EPE/data/embeddings-4"

    cv_trainer = CrossValidationTrainer(
        model_name='Lrtf', 
        n_splits=10, 
        batch_size=128, 
        epochs=300,
        learning_rate=0.00001,
        weight_decay=1e-5,
        emdedding_model=embedding_model,
        basedir=basedir,
        emdedding_model_path=emdedding_model_path
    )
    
    cv_metrics_list, cv_summary = cv_trainer.run_cross_validation()
    
    print("\n===== 交叉验证结果 =====")  
    print("每折指标:")
    for fold_metrics in cv_metrics_list:
        print(f"Fold {fold_metrics['fold']} 指标:")
        for metric, value in fold_metrics.items():
            if metric != 'fold':
                print(f"  {metric}: {value:.4f}")
    
    print("\n平均指标:")
    # json_dict={}
    for metric, value in cv_summary.items():
        print(f"{metric}: {value:.4f}")
        # json_dict["metric"]=value

    results_df = pd.DataFrame(cv_metrics_list)  
    results_df["embedding_model"] = embedding_model.split(".")[0]
    result_path = os.path.join( basedir, f"checkpoints{temp}", embedding_model.split(".")[0] + ".csv" ) 
    results_df.to_csv(result_path, index=False)

if __name__ == "__main__":
    main()