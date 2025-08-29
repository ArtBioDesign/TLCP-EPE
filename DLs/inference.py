import sys
import time
import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import torch.utils.data as Data
from importlib import import_module
import torch.nn as nn
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from utils import DataProcessor, BiolmDataSet, ModelEvaluator

def count_parameters(model):
    """计算模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'Total Parameters': total_params,
        'Trainable Parameters': trainable_params,
        'Non-trainable Parameters': total_params - trainable_params
    }

# 加载数据
data_processor = DataProcessor()

basedir = "/hpcfs/fhome/yangchh/workdir/self/TLCP-EPE/test/test_data/embeddings_4"   #/prot_t5_xl_uniref50.pkl Finetune_CaLM+ProtT5 huggingface_calm
finetune_model_name = "Finetune_CaLM+ProtT5.pkl"

features, labels = data_processor.load_data(os.path.join(basedir, finetune_model_name))
X_scaled, y_scaled = data_processor.preprocess_data(features, labels)
X_test, y_test = X_scaled, y_scaled

# 创建测试数据集
test_dataset = BiolmDataSet(X_test, y_test)
batch_size = 128
test_iter = Data.DataLoader(test_dataset, batch_size=batch_size)

# 模型设置
model_name = 'Lrtf'    
x = import_module(f'models.{model_name}')
config = x.Config()

# 加载十折交叉验证的模型
model_dir = '/hpcfs/fhome/yangchh/workdir/self/TLCP-EPE/DLs/checkpoints'
models = [] 
for fold in range(1, 11):
    model_path = os.path.join(model_dir, f'fold_{fold}_best_model.pt')
    model = x.Model(config).to(config.device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    models.append(model)

# 打印第一个模型的参数量作为示例
params_info = count_parameters(models[0])
print("\nModel Parameters Summary (Example for Fold 1):")
print("-" * 50)
print(f"Total Parameters: {params_info['Total Parameters']:,}")
print(f"Trainable Parameters: {params_info['Trainable Parameters']:,}")
print(f"Non-trainable Parameters: {params_info['Non-trainable Parameters']:,}")
print("-" * 50)

# 初始化 ModelEvaluator
evaluator = ModelEvaluator()

def evaluate(models, test_iter):
    all_probabilities = []
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for test_x, test_y in test_iter:
            test_x = test_x.cuda().float()
            test_y = test_y.cuda().long()
            
            # 对每个模型进行预测
            fold_probabilities = []
            for model in models:
                outputs = model(test_x)
                probabilities = F.softmax(outputs, dim=1)[:, 1]  # 预测为高表达的概率（类别1）
                fold_probabilities.append(probabilities.cpu().numpy())
            
            # 计算每个样本的平均概率和方差
            fold_probabilities = np.array(fold_probabilities)  # 形状为 (10, batch_size)
            mean_probabilities = np.mean(fold_probabilities, axis=0)
            var_probabilities = np.var(fold_probabilities, axis=0)
            
            # 将平均概率转换为二分类预测（阈值 0.5）
            predictions = (mean_probabilities >= 0.5).astype(int)
            
            all_probabilities.append({
                'mean': mean_probabilities,
                'var': var_probabilities
            })
            all_predictions.extend(predictions)
            all_labels.extend(test_y.cpu().numpy())
    
    # 合并所有批次的概率、预测和标签
    mean_probs = np.concatenate([p['mean'] for p in all_probabilities])
    var_probs = np.concatenate([p['var'] for p in all_probabilities])
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    
    # 计算低表达样本（标签为0）预测为高表达的平均概率
    low_expr_mask = all_labels == 0
    avg_prob_low_to_high = np.mean(mean_probs[low_expr_mask]) if np.any(low_expr_mask) else 0
    
    # 计算高表达样本（标签为1）预测为高表达的平均概率
    high_expr_mask = all_labels == 1
    avg_prob_high_to_high = np.mean(mean_probs[high_expr_mask]) if np.any(high_expr_mask) else 0
    
    if len(set(all_labels)) != 1:  
        # 使用 ModelEvaluator 计算评估指标
        metrics_results = evaluator.evaluate_model(
            y_pred=all_predictions,
            y_test=all_labels,
            y_pred_prob_fold=mean_probs,  # 用于 AUC 计算
            task_type="bc"
        )

        # 打印评估指标
        print("\nModel Performance Metrics:")
        print("-" * 50)
        for metric, value in metrics_results.items():
            print(f"{metric}: {value:.4f}")
        
    # 打印结果
    print("\nExpression Level Prediction Analysis:")
    print("-" * 50)
    print(f"Average probability of predicting high expression for low expression samples: {avg_prob_low_to_high:.4f}")
    print(f"Average probability of predicting high expression for high expression samples: {avg_prob_high_to_high:.4f}")
    
    # 打印样本数量
    low_expr_count = np.sum(low_expr_mask)
    high_expr_count = np.sum(high_expr_mask)
    print("\nSample Distribution:")
    print(f"Number of low expression samples (0): {low_expr_count}")
    print(f"Number of high expression samples (1): {high_expr_count}")
    
   
    
    # 返回结果
    return {
        'mean_probabilities': mean_probs,
        'var_probabilities': var_probs,
        'labels': all_labels,
        'predictions': all_predictions,
        'avg_prob_low_to_high': avg_prob_low_to_high,
        'avg_prob_high_to_high': avg_prob_high_to_high,
        'metrics': metrics_results if len(set(all_labels))!=1 else None
    }

# 执行测试评估
results = evaluate(models, test_iter)

pd.DataFrame( [results["metrics"]] ).to_csv( os.path.join( basedir, finetune_model_name[:-4]+".csv" ),index=False )


# 保存测试结果
results_df = pd.DataFrame({
    'True_Labels': results['labels'],
    'Predictions': results['predictions'],
    'Mean_Probability': results['mean_probabilities'],
    'Variance': results['var_probabilities']
})
results_df.to_csv('test_results_with_variance.csv', index=False)
print("\nTest results saved to 'test_results_with_variance.csv'")


