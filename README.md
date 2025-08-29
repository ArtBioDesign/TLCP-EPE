  
# TLCP-EPE
## Project Introduction  
**TLCP-EPE** This study proposes a TLCP-EPE framework, which is capable of qualitatively predicting protein expression levels in E. coli from the perspectives of proteins, and the joint perception of codons and proteins..
![TLCP-EPE](https://github.com/editSeqDesign/AutoPMD/blob/main/img/home.png) 

## Installation
### python packages
We suggest using Python 3.8 for TLCP-EPE.

```shell
pip install -r requirements.txt

```

## Usage & Example

**Input:**
- **Step 1:** 下载TLCP-EPE checkpoint到 DLs/BiGRU-Attention-MLP目录下；将需要输入的蛋白序列或CDS序列放到data目录下.
- **Step 2:** 基于PLMs获取序列表示
    ```shell
        python FMs/do_embedding.py -p prot_t5_xl_uniref50  -pmt protein -dp data -fn test.csv 
    ```
- **Step 3:** 基于TLCP-EPE的checkpoint预测
     ```shell
        python DLs/test.py 
    ```

