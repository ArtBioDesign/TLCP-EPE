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
from torch.utils.data import DataLoader
from utils import DLMs_DataProcessor, BiolmDataSet, ModelEvaluator
from config import CALM_PLM_CONFIG, T5_PLM_CONFIG
from extract_embedding import EmbeddingExtractor


def count_parameters(model):
    """计算模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'Total Parameters': total_params,
        'Trainable Parameters': trainable_params,
        'Non-trainable Parameters': total_params - trainable_params
    }


def cancat_codon_protein_representation(df1, df2):
    df = pd.DataFrame(columns=["id", "embedding", "label"])
    for i in range(len(df1)):
        df.loc[i, "id"] = df1.loc[i, "id"]
        df.loc[i, "embedding"] = np.hstack((df1.loc[i, "embedding"], df2.loc[i, "embedding"]))
        df.loc[i, "label"] = df1.loc[i, "label"]
    return df


def custom_collate_fn(batch):
    features, labels = zip(*batch)
    features = torch.stack(features, dim=0)
    labels = torch.stack(labels, dim=0)
    return features, labels


def main():
    parser = argparse.ArgumentParser(description="Evaluate ensemble models on test data.")
    parser.add_argument('--model_dir', type=str, required=True, help='Directory of saved models (fold_*.pt)')
    parser.add_argument('--model_name', type=str, required=True, help='Model name (e.g., BiGRU_Attention_MLP)')
    parser.add_argument('--output_file', type=str, default='result.csv', help='Output CSV file')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for DataLoader')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on (cuda or cpu)')
    parser.add_argument('--print_model_params', action='store_true', help='Print model parameter count')
    args = parser.parse_args()

    # 加载嵌入
    calm_embeddings = EmbeddingExtractor(CALM_PLM_CONFIG)._run()
    t5_embeddings = EmbeddingExtractor(T5_PLM_CONFIG)._run()
    Calm_t5 = cancat_codon_protein_representation(calm_embeddings, t5_embeddings)

    # 数据预处理
    data_processor = DLMs_DataProcessor()
    features = pd.DataFrame(Calm_t5.pop('embedding').to_list(), index=Calm_t5.index)
    labels = Calm_t5["label"]
    X_scaled, y_scaled = data_processor.preprocess_data(features, labels)
    X_test, y_test = X_scaled, y_scaled

    # 创建测试数据集
    test_dataset = BiolmDataSet(X_test, y_test)
    test_iter = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=custom_collate_fn)

    # 加载模型
    x = import_module(f'models.{args.model_name}')
    config = x.Config()
    config.device = torch.device(args.device)

    models = []
    for fold in range(1, 11):
        model_path = os.path.join(args.model_dir, f'fold_{fold}_best_model.pt')
        model = x.Model(config).to(config.device)
        model.load_state_dict(torch.load(model_path, map_location=config.device))
        model.eval()
        models.append(model)

    # 打印模型参数量
    if args.print_model_params:
        params_info = count_parameters(models[0])
        print("\nModel Parameters Summary (Example for Fold 1):")
        print("-" * 50)
        print(f"Total Parameters: {params_info['Total Parameters']:,}")
        print(f"Trainable Parameters: {params_info['Trainable Parameters']:,}")
        print(f"Non-trainable Parameters: {params_info['Non-trainable Parameters']:,}")
        print("-" * 50)

    # 初始化评估器
    evaluator = ModelEvaluator()

    def evaluate(models, test_iter):
        all_probabilities = []
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for test_x, test_y in test_iter:
                test_x = test_x.to(config.device).float()
                test_y = test_y.to(config.device).long()

                fold_probabilities = []
                for model in models:
                    outputs = model(test_x)
                    probabilities = F.softmax(outputs, dim=1)[:, 1]
                    fold_probabilities.append(probabilities.cpu().numpy())

                fold_probabilities = np.array(fold_probabilities)
                mean_probabilities = np.mean(fold_probabilities, axis=0)
                var_probabilities = np.var(fold_probabilities, axis=0)
                predictions = (mean_probabilities >= 0.5).astype(int)

                all_probabilities.append({'mean': mean_probabilities, 'var': var_probabilities})
                all_predictions.extend(predictions)
                all_labels.extend(test_y.cpu().numpy())

        mean_probs = np.concatenate([p['mean'] for p in all_probabilities])
        var_probs = np.concatenate([p['var'] for p in all_probabilities])
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)

        low_expr_mask = all_labels == 0
        high_expr_mask = all_labels == 1

        avg_prob_low_to_high = np.mean(mean_probs[low_expr_mask]) if np.any(low_expr_mask) else 0
        avg_prob_high_to_high = np.mean(mean_probs[high_expr_mask]) if np.any(high_expr_mask) else 0

        metrics_results = None
        if len(set(all_labels)) != 1:
            metrics_results = evaluator.evaluate_model(
                y_pred=all_predictions,
                y_test=all_labels,
                y_pred_prob_fold=mean_probs,
                task_type="bc"
            )

            print("\nModel Performance Metrics:")
            print("-" * 50)
            for metric, value in metrics_results.items():
                print(f"{metric}: {value:.4f}")

        print("\nExpression Level Prediction Analysis:")
        print("-" * 50)
        print(f"Average probability of predicting high expression for low expression samples: {avg_prob_low_to_high:.4f}")
        print(f"Average probability of predicting high expression for high expression samples: {avg_prob_high_to_high:.4f}")

        low_expr_count = np.sum(low_expr_mask)
        high_expr_count = np.sum(high_expr_mask)
        print("\nSample Distribution:")
        print(f"Number of low expression samples (0): {low_expr_count}")
        print(f"Number of high expression samples (1): {high_expr_count}")

        return {
            'mean_probabilities': mean_probs,
            'var_probabilities': var_probs,
            'labels': all_labels,
            'predictions': all_predictions,
            'avg_prob_low_to_high': avg_prob_low_to_high,
            'avg_prob_high_to_high': avg_prob_high_to_high,
            'metrics': metrics_results
        }

    results = evaluate(models, test_iter)

    # 保存结果
    results_df = pd.DataFrame({
        'id': calm_embeddings["id"],
        'True_Labels': results['labels'],
        'Predictions': results['predictions'],
        'Mean_Probability': results['mean_probabilities'],
        'Variance': results['var_probabilities']
    })

    results_df.to_csv(args.output_file, index=False)
    print(f"\nTest results saved to '{args.output_file}'")


if __name__ == "__main__":
    main()


