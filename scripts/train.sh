# 原始单GPU训练命令
python3 -u -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --master_port=10000  train.py --cfg-path lavis/projects/blip2/train/fga_blip2.yaml

# 多GPU训练命令 (使用2个GPU)
# python3 -u -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=10000  train.py --cfg-path lavis/projects/blip2/train/fga_blip2.yaml

# 多GPU训练命令 (使用4个GPU)
# python3 -u -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port=10000  train.py --cfg-path lavis/projects/blip2/train/fga_blip2.yaml

# 多GPU训练命令 (使用全部可用GPU)
# 注意：使用多GPU时，实际批次大小 = batch_size_train × GPU数量
# 如果总批次大小过大，可能需要相应调整学习率
# python3 -u -m torch.distributed.run --nnodes=1 --nproc_per_node=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) --master_port=10000  train.py --cfg-path lavis/projects/blip2/train/fga_blip2.yaml