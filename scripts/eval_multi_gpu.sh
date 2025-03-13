#!/bin/bash

# 使用多GPU评估脚本
# 参数说明:
# --num_gpus: 使用的GPU数量
# --batch_size: 每个GPU的批处理大小
# --model_path: 模型路径
# --save_path: 结果保存路径
# --data_file: 数据文件路径
# --dataset_dir: 数据集目录

# 评估最新训练的模型
python eval_multi_gpu.py \
  --model_path /teamspace/studios/this_studio/lavis/output/FGA-BLIP2/[新训练目录]/checkpoint_0.pth \
  --save_path results/result_new_balanced_multi_gpu.json \
  --data_file dataset/val_split_05.json \
  --dataset_dir dataset/images/ \
  --batch_size 16 \
  --num_gpus 4

# 计算评估指标
python calculate_metrics.py \
  --result_file results/result_new_balanced_multi_gpu.json \
  --ground_truth dataset/val_split_05.json \
  --output results/metrics_new_balanced_multi_gpu.json
