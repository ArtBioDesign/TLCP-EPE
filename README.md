# TLCP-EPE: Two-Level Codon-Protein Expression Prediction Framework

## 🧬 Project Introduction

**TLCP-EPE** is a deep learning framework designed to **predict protein expression levels in *E. coli*** from both **codon usage** and **protein sequence** perspectives. The model integrates protein language models (PLMs) and codon-aware representations with a BiGRU-Attention-MLP architecture for accurate and interpretable predictions.  
![TLCP-EPE](https://github.com/ArtBioDesign/TLCP-EPE/blob/main/image/TLCP-EPE.PNG) 

## Installation
### python packages
We suggest using Python 3.8 for TLCP-EPE.

```shell
pip install -r requirements.txt

```

## Usage & Example

- **Step 1:** 
    Download the following model checkpoints and place them in the appropriate directories:
    TLCP-EPE Main Model: 
        ~/TLCP-EPE/DLs/BiGRU-Attention-MLP/TLCP-EPE-checkpoints/;
    Protein PLM (ProtT5) & Codon PLM (CALM) Fine-tuned Weights:
        Place under ~/TLCP-EPE/data/saved_models_*/ as specified in config.py
- **Step 2:** 
    ```shell
    T5_PLM_CONFIG = {
            'pretrained_model': "Rostlab/prot_t5_xl_uniref50",
            'pretrained_model_type': "protein",
            'finetuned_params_path': "~/TLCP-EPE/data/saved_models_prot_t5_xl_uniref50_lora_fold_4/fold_4_finetuned_params.pth",
            'data_path': "~/TLCP-EPE/data",
            'batch_size': 4, 
            'max_seq_length': None,
            'file_name': "sample.csv",
            'task': "do_finetune",
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
        }

    CALM_PLM_CONFIG = {
                'pretrained_model': "~/TLCP-EPE/DLs/pretrain_codon/huggingface_calm",
                'pretrained_model_type': "codon",
                'finetuned_params_path': "~/TLCP-EPE/data/saved_models_huggingface_calm_lora_fold_4/fold_4_finetuned_params.pth",
                'data_path': "~/TLCP-EPE/data",
                'batch_size': 16, 
                'max_seq_length': None,
                'file_name': 'sample.csv',
                'task': 'do_finetune',
                'device' : torch.device('cuda' if torch.cuda.is_available() else 'cpu')     
            }
    ```

- **Step 2:** 基于TLCP-EPE的checkpoint预测
    ```shell
        python inference.py \
            --model_dir BiGRU-Attention-MLP/TLCP-EPE-checkpoints \
            --model_name BiGRU_Attention_MLP \
            --output_file test_results_with_variance.csv \
            --batch_size 16 \
            --device cuda \
            --print_model_params    
    ```
- **Output:**
    - `~/TLCP-EPE/test_results_with_variance.csv` 

