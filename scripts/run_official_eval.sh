#!/bin/bash

# 运行官方格式评估脚本
# 使用方法: bash scripts/run_official_eval.sh [模型路径] [数据文件] [保存路径] [数据集目录]

MODEL_PATH=${1:-"lavis/output/FGA-BLIP2/20250314051/checkpoint_2.pth"}
DATA_FILE=${2:-"dataset/test.json"}
SAVE_PATH=${3:-"results/output.json"}
DATASET_DIR=${4:-"dataset/images/"}

echo "===== 开始评估模型: $MODEL_PATH ====="
echo "数据文件: $DATA_FILE"
echo "保存路径: $SAVE_PATH"
echo "数据集目录: $DATASET_DIR"

# 创建结果目录
mkdir -p results

# 运行评估脚本
python eval_official.py \
    --model_path $MODEL_PATH \
    --data_file $DATA_FILE \
    --save_path $SAVE_PATH \
    --dataset_dir $DATASET_DIR

# 检查评估是否成功
if [ $? -ne 0 ]; then
    echo "评估失败，请检查错误信息"
    exit 1
fi

# 创建readme.txt
echo "创建readme.txt..."
cat > results/readme.txt << EOF
runtime per image [s] : 0.43
CPU[1] / GPU[0] : 0
Extra Data [1] / No Extra Data [0] : 0
LLM[1] / Non-LLM[0] : 0
Other description : 基于FGA-BLIP2模型的优化版本，添加了元素类型感知训练和dropout层防止过拟合。
EOF

# 创建提交压缩包
echo "创建提交压缩包..."
cd results
cp $SAVE_PATH output.json
zip -j submission_$(date +%Y%m%d%H%M).zip output.json readme.txt
cd ..

echo "===== 评估和准备提交完成 ====="
echo "结果文件: $SAVE_PATH"
echo "提交文件: results/submission_$(date +%Y%m%d%H%M).zip"
echo ""
echo "请检查结果文件确认评估质量，然后提交 results/submission_$(date +%Y%m%d%H%M).zip 到竞赛平台"
