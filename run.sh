#!/bin/bash

# 检查是否传入了序列类型参数
if [ $# -eq 0 ]; then
    echo "用法: $0 <sequence_type>"
    echo "例如: $0 DNA"
    echo "支持的类型: DNA, RNA, Protein 等（根据模型实际支持情况）"
    exit 1
fi

# 获取传入的序列类型
SEQUENCE_TYPE="$1"

# 定义基础路径
BASE_PATH="/hpcfs/fhome/yangchh/tools_deployed/tlcp-epe"
DATA_PATH="${BASE_PATH}/data"

# 创建输出目录
mkdir -p "${DATA_PATH}/output"

# 构建输入和输出文件路径（可根据 sequence_type 动态调整）
INPUT_FILE="${DATA_PATH}/input/${SEQUENCE_TYPE}.fasta"
OUTPUT_FILE="${DATA_PATH}/output/results_${SEQUENCE_TYPE}.csv"

# 检查输入文件是否存在
if [ ! -f "$INPUT_FILE" ]; then
    echo "错误: 输入文件不存在: $INPUT_FILE"
    exit 1
fi

# 执行推理命令
/hpcfs/fhome/yangchh/software/miniforge3/envs/bioseq/bin/python "${BASE_PATH}/TLCP-EPE/src/inference.py" \
    --sequence_type "${SEQUENCE_TYPE}" \
    --workdir_data "${DATA_PATH}" \
    --input_file "${INPUT_FILE}" \
    --output_file "${OUTPUT_FILE}" \
    --dlm_checkpoints "${DATA_PATH}/dlm_checkpoints" \
    --protein_PLM "Rostlab/prot_t5_xl_uniref50" \
    --protein_PLM_finetune_params_path "${DATA_PATH}/plms/ProtT5/fold_4_finetuned_params.pth" \
    --codon_PLM "xiaoyangch/HCaLM" \
    --codon_PLM_finetune_params_path "${DATA_PATH}/plms/HCaLM/fold_4_finetuned_params.pth"

echo "执行完成，结果保存在: ${OUTPUT_FILE}"
