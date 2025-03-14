# 默认评估脚本
python3 eval.py
# 编辑脚本 评估训练后的模型
echo 'python3 eval.py --model_path output/FGA-BLIP2/20250312140/checkpoint_4.pth --save_path results/result_epoch4.json' > scripts/eval.sh
python3 eval.py --model_path /teamspace/studios/this_studio/lavis/output/FGA-BLIP2/20250312140/checkpoint_4.pth --save_path results/result_epoch4.json --batch_size 32
python eval_optimized.py --model_path /teamspace/studios/this_studio/lavis/output/FGA-BLIP2/20250312140/checkpoint_4.pth --save_path results/result_epoch4.json --batch_size 32
python eval_compatible.py --model_path lavis/output/FGA-BLIP2/20250314051/checkpoint_2.pth --batch_size 32