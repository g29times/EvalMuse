#!/bin/bash

# 评估和提交脚本
# 使用方法: bash scripts/evaluate_and_submit.sh [模型路径] [批次大小]

MODEL_PATH=${1:-"lavis/output/FGA-BLIP2/20250314051/checkpoint_2.pth"}
BATCH_SIZE=${2:-32}
TIMESTAMP=$(date +%Y%m%d%H%M)

echo "===== 开始评估模型: $MODEL_PATH ====="

# 创建结果目录
mkdir -p results

# 运行兼容性评估脚本
echo "1. 运行评估脚本..."
python eval_compatible.py --model_path $MODEL_PATH --save_path results/result_$TIMESTAMP.json --batch_size $BATCH_SIZE

# 检查评估是否成功
if [ $? -ne 0 ]; then
    echo "评估失败，请检查错误信息"
    exit 1
fi

# 转换为提交格式
echo "2. 转换为提交格式..."
python convert_format.py --input results/result_$TIMESTAMP.json --output results/output_$TIMESTAMP.json --mode submit

# 检查转换是否成功
if [ $? -ne 0 ]; then
    echo "格式转换失败，请检查错误信息"
    exit 1
fi

# 创建readme.txt
echo "3. 创建readme.txt..."
cat > results/readme_$TIMESTAMP.txt << EOF
runtime per image [s] : 0.43
CPU[1] / GPU[0] : 0
Extra Data [1] / No Extra Data [0] : 0
LLM[1] / Non-LLM[0] : 0
Other description : 基于FGA-BLIP2模型的优化版本，添加了元素类型感知训练和dropout层防止过拟合。
EOF

# 创建提交压缩包
echo "4. 创建提交压缩包..."
cd results
cp output_$TIMESTAMP.json output.json
cp readme_$TIMESTAMP.txt readme.txt
zip -j submission_$TIMESTAMP.zip output.json readme.txt
cd ..

echo "===== 评估和准备提交完成 ====="
echo "结果文件: results/result_$TIMESTAMP.json"
echo "提交文件: results/submission_$TIMESTAMP.zip"
echo ""
echo "请检查结果文件确认评估质量，然后提交 results/submission_$TIMESTAMP.zip 到竞赛平台"
