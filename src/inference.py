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
from config import PLM_Config
from extract_embedding import EmbeddingExtractor
from Bio import SeqIO

# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)

def fasta_to_dataframe(fasta_path, workdir_data = ".", sequence_type="dna"):

    # Initialize lists to store the data
    records = []
    
    # Parse the FASTA file
    with open(fasta_path, 'r') as handle:
        for record in SeqIO.parse(handle, 'fasta'):
            records.append({
                'name': record.id,
                'sequence': str(record.seq),
                "sequence_type": sequence_type
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(records)

    write_path = os.path.join( workdir_data, "sample.csv" )

    df.to_csv( write_path, index=False )
    
    return df

def cancat_codon_protein_representation(df1, df2):
    df = pd.DataFrame(columns=["id", "embedding", "label"])
    for i in range(len(df1)):
        df.loc[i, "id"] = df1.loc[i, "id"]
        df.loc[i, "embedding"] = np.hstack((df1.loc[i, "embedding"], df2.loc[i, "embedding"]))
    return df

def main():
    parser = argparse.ArgumentParser(description="Evaluate ensemble models on test data.")
    parser.add_argument('--sequence_type', type=str, required=True, help='DNA or Protein')

    parser.add_argument('--workdir_data', type=str, default="/hpcfs/fhome/yangchh/tools_deployed/tlcp-epe/data", help='workdir data path')

    parser.add_argument('--input_file', type=str, required=True, default="/hpcfs/fhome/yangchh/tools_deployed/tlcp-epe/data/input/DNA.fasta", help='input path')
    parser.add_argument('--output_file', type=str, default='/hpcfs/fhome/yangchh/tools_deployed/tlcp-epe/data/output/results.csv', help='output path')
    parser.add_argument('--dlm_checkpoints', type=str, default="/hpcfs/fhome/yangchh/tools_deployed/tlcp-epe/data/dlm_checkpoints", help='deep model path')\
    
    parser.add_argument('--protein_PLM', type=str, default="/hpcfs/fhome/yangchh/tools_deployed/tlcp-epe/data/plms/ProtT5/snapshots/973be27c52ee6474de9c945952a8008aeb2a1a73", help='Protein plm params path')
    parser.add_argument('--protein_PLM_finetune_params_path', type=str, default="/hpcfs/fhome/yangchh/tools_deployed/tlcp-epe/data/plms/ProtT5/fold_4_finetuned_params.pth", help='Protein plm finetune params path')
    parser.add_argument('--codon_PLM', type=str, default="/hpcfs/fhome/yangchh/tools_deployed/tlcp-epe/data/plms/HCaLM/snapshots/4eec0b9320b2ae79c858c107346fd28db57eaa44", help='Codon plm params path')
    parser.add_argument('--codon_PLM_finetune_params_path', type=str, default="/hpcfs/fhome/yangchh/tools_deployed/tlcp-epe/data/plms/HCaLM/fold_4_finetuned_params.pth", help='codon plm finetune params path')

    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for DataLoader')
    args = parser.parse_args()

    sequence_type = args.sequence_type

    fasta_to_dataframe( args.input_file, workdir_data=args.workdir_data,  sequence_type=sequence_type.lower() )

    model_name = "TLCP-EPE" if sequence_type == "DNA" else "TLP-EPE"
    model_dir = os.path.join( args.dlm_checkpoints, model_name )

    PLM_Config.T5_PLM_CONFIG["data_path"] = args.workdir_data
    PLM_Config.CALM_PLM_CONFIG["data_path"] = args.workdir_data

    PLM_Config.T5_PLM_CONFIG["pretrained_model"] = args.protein_PLM
    PLM_Config.T5_PLM_CONFIG["finetuned_params_path"] = args.protein_PLM_finetune_params_path
    
    PLM_Config.CALM_PLM_CONFIG["pretrained_model"] = args.codon_PLM
    PLM_Config.CALM_PLM_CONFIG["finetuned_params_path"] = args.codon_PLM_finetune_params_path
    
    if model_name == "TLCP-EPE":
        calm_embeddings = EmbeddingExtractor(PLM_Config.CALM_PLM_CONFIG)._run()
        t5_embeddings = EmbeddingExtractor(PLM_Config.T5_PLM_CONFIG)._run()
        Calm_t5 = cancat_codon_protein_representation(calm_embeddings, t5_embeddings)
    elif model_name == "TLP-EPE": 
        t5_embeddings = EmbeddingExtractor(PLM_Config.T5_PLM_CONFIG)._run()
        Calm_t5 = t5_embeddings

    data_processor = DLMs_DataProcessor()
    features = pd.DataFrame(Calm_t5.pop('embedding').to_list(), index=Calm_t5.index)
    X_scaled = data_processor.preprocess_data(features)
    X_test = X_scaled

    test_dataset = BiolmDataSet(X_test)
    test_iter = DataLoader(test_dataset, batch_size=args.batch_size)

    x = import_module(f'models.model')
    embed = 1792 if model_name == "TLCP-EPE" else 1024
    config = x.Config()
    config.embed = embed
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    models = []
    for fold in range(1, 11):
        model_path = os.path.join(model_dir, f'fold_{fold}_best_model.pt')
        model = x.Model(config).to(config.device)
        model.load_state_dict(torch.load(model_path, map_location=config.device))
        model.eval()
        models.append(model)

    def run(models, test_iter):
        all_probabilities = []
        all_predictions = []

        with torch.no_grad():
            for test_x in test_iter:
                test_x = test_x.to(config.device).float()
                
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
               
        mean_probs = np.concatenate([p['mean'] for p in all_probabilities])
        var_probs = np.concatenate([p['var'] for p in all_probabilities])
        all_predictions = np.array(all_predictions)

        return {
            'mean_probabilities': mean_probs,
            'var_probabilities': var_probs,
            'predictions': all_predictions,
        }

    results = run(models, test_iter)

    # 保存结果
    results_df = pd.DataFrame({
        'name': t5_embeddings["id"],
        'Predictions': results['predictions'],
        'Score': results['mean_probabilities'],
    })

    os.makedirs( os.path.dirname( args.output_file ), exist_ok=True)
    
    results_df.to_csv(args.output_file, index=False)
    print(f"\nTest results saved to '{args.output_file}'")

if __name__ == "__main__":
    main()