# TLCP-EPE: Two-Level Codon-Protein Expression Prediction Framework

## 🧬 Project Introduction

**TLCP-EPE** is a deep learning framework designed to **predict protein expression levels in *E. coli*** from both **codon usage** and **protein sequence** perspectives. The model integrates protein language models (PLMs) and codon-aware representations with a BiGRU-MLP architecture for accurate and interpretable predictions.  
![TLCP-EPE](https://github.com/ArtBioDesign/TLCP-EPE/blob/master/TLCP-EPE.PNG) 

## Installation
### python packages
We suggest using Python 3.8 for TLCP-EPE.

```shell
pip install -r requirements.txt

```

## Usage & Example
- **Step 1:** 
    Download the data/ folder from Zenodo (https://zenodo.org/uploads/17011129) and place it in the root directory of the project

- **Step 2:** 
    Open run.sh and adjust paths or parameters 

- **Step 3:** 
    ```shell
    bash run.sh DNA
    ```
- **Output:**
    - `~/TLCP-EPE/data/test_results_with_variance.csv` 

