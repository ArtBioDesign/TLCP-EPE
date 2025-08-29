
# TLCP-EPE
## Project Introduction  
**TLCP-EPE** This study proposes a TLCP-EPE framework, which is capable of qualitatively predicting protein expression levels in E. coli from the perspectives of proteins, and the joint perception of codons and proteins..
![TLCP-EPE](https://github.com/ArtBioDesign/TLCP-EPE/blob/main/TLCP-EPE.PNG) 

## Installation
### python packages
We suggest using Python 3.8 for TLCP-EPE.

```shell
pip install -r requirements.txt

```

## Usage & Example

**Input:**
- **Step 1:** Download the TLCP-EPE checkpoint to the DLs/BiGRU-Attention-MLP directory and place the protein sequence or CDS sequence to be input into the data directory..
- **Step 2:** Obtaining sequence representation based on PLMs
    ```shell
        python FMs/do_embedding.py -p prot_t5_xl_uniref50  -pmt protein -dp data -fn test.csv 
    ```
- **Step 3:** Checkpoint prediction based on TLCP-EPE
    ```shell
        python DLs/test.py 
    ```
    
