import torch
import json
import csv
from transformers import BertTokenizer
from tqdm import tqdm
from lavis.models import load_model_and_preprocess, load_model
import os
from PIL import Image
from utils import compute_metrics, load_data
import argparse
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# 创建数据集类
class EvalDataset(Dataset):
    def __init__(self, data, dataset_dir, vis_processors, text_processors, tokenizer):
        self.data = data
        self.dataset_dir = dataset_dir
        self.vis_processors = vis_processors
        self.text_processors = text_processors
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        elements = item['element_score'].keys()
        prompt = item['prompt']
        
        image_path = os.path.join(self.dataset_dir, item['img_path'])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processors["eval"](image)
        
        processed_prompt = self.text_processors["eval"](prompt)
        prompt_ids = self.tokenizer(prompt).input_ids
        
        return {
            "image": image,
            "prompt": processed_prompt,
            "prompt_ids": prompt_ids,
            "elements": list(elements),
            "item": item
        }

def binarize(lst):
    # 将列表中的元素按 0.5 阈值进行二值化
    return [1 if x >= 0.5 else 0 for x in lst]

def calculate_accuracy(true_list, pred_list):
    # 先二值化两个列表
    true_bin = binarize(true_list)
    pred_bin = binarize(pred_list)
    
    # 计算相同元素的数量
    correct = sum([1 for t, p in zip(true_bin, pred_bin) if t == p])
    loss = torch.mean(torch.abs(torch.tensor(true_list)- torch.tensor(pred_list)))
    # 计算准确率
    accuracy = correct / len(true_list)
    return accuracy, loss

def is_sublist(lst1, lst2):
    return str(lst1)[1:-1] in str(lst2)[1:-1]

def get_index(list1, list2):
    len_list1 = len(list1)
    len_list2 = len(list2)
    for i in range(len_list2 - len_list1 + 1):
        if list2[i:i + len_list1] == list1:
            return i
    return 0

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def process_batch(batch, model, device, tokenizer):
    results = []
    images = batch["image"].to(device)
    prompts = batch["prompt"]
    
    # 批量推理
    with torch.no_grad():
        alignment_scores, scores_batch = model.element_score(images, prompts)
    
    # 处理每个样本的结果
    for i in range(len(batch["item"])):
        item = batch["item"][i]
        elements = batch["elements"][i]
        prompt_ids = batch["prompt_ids"][i]
        alignment_score = alignment_scores[i]
        scores = scores_batch[i]
        
        # 处理元素评分
        elements_score = dict()
        for element in elements:
            element_ = element.rpartition('(')[0]
            element_ids = tokenizer(element_).input_ids[1:-1]
            
            idx = get_index(element_ids, prompt_ids)
            if idx:
                # 确保掩码长度与分数张量匹配
                scores_len = scores.shape[0]
                
                # 检查掩码是否会超出分数张量的长度
                if idx >= scores_len:
                    elements_score[element] = 0
                    continue
                
                # 调整掩码长度以匹配分数张量
                mask = [0] * scores_len
                end_idx = min(idx + len(element_ids), scores_len)
                mask[idx:end_idx] = [1] * (end_idx - idx)
                
                mask = torch.tensor(mask).to(device)
                if mask.sum() > 0:  # 避免除零错误
                    elements_score[element] = ((scores * mask).sum() / mask.sum()).item()
                else:
                    elements_score[element] = 0
            else:
                elements_score[element] = 0
        
        item_copy = item.copy()
        item_copy['score_result'] = alignment_score.item() if isinstance(alignment_score, torch.Tensor) else alignment_score
        item_copy['element_result'] = elements_score
        results.append(item_copy)
    
    return results

def eval_process(rank, world_size, args):
    # 初始化分布式环境
    setup(rank, world_size)
    
    # 设置设备
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    # 加载分词器
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side='right')
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    
    # 加载数据
    data = load_data(args.data_file, 'json')
    
    # 加载模型和处理器
    model, vis_processors, text_processors = load_model_and_preprocess("fga_blip2", "coco", device=device, is_eval=True)
    model.load_checkpoint(args.model_path)
    model.eval()
    
    # 创建数据集和数据加载器
    dataset = EvalDataset(data, args.dataset_dir, vis_processors, text_processors, tokenizer)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # 评估
    local_results = []
    for batch in tqdm(dataloader, desc=f"GPU {rank} processing", disable=rank != 0):
        batch_results = process_batch(batch, model, device, tokenizer)
        local_results.extend(batch_results)
    
    # 收集所有进程的结果
    all_results = [None for _ in range(world_size)]
    dist.all_gather_object(all_results, local_results)
    
    # 主进程保存结果
    if rank == 0:
        # 合并结果
        merged_results = []
        for results in all_results:
            merged_results.extend(results)
        
        # 按原始数据顺序排序
        result_dict = {item['prompt_id']: item for item in merged_results}
        ordered_results = [result_dict[item['prompt_id']] for item in data if item['prompt_id'] in result_dict]
        
        # 保存结果
        with open(args.save_path, 'w', newline='', encoding='utf-8') as file:
            json.dump(ordered_results, file, ensure_ascii=False, indent=4)
        
        print(f"评估完成，结果已保存到 {args.save_path}")
    
    # 清理
    cleanup()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default='dataset/val_split.json')
    parser.add_argument('--save_path', type=str, default='results/result_multi_gpu.json')
    parser.add_argument('--model_path', type=str, default='checkpoints/best.pth')
    parser.add_argument('--dataset_dir', type=str, default='dataset/images/')
    parser.add_argument('--batch_size', type=int, default=16, help='每个GPU的批处理大小')
    parser.add_argument('--num_gpus', type=int, default=4, help='使用的GPU数量')
    args = parser.parse_args()
    
    # 检测可用GPU数量
    available_gpus = torch.cuda.device_count()
    num_gpus = min(args.num_gpus, available_gpus)
    
    if num_gpus > 1:
        print(f"使用 {num_gpus} 个GPU进行评估")
        mp.spawn(eval_process, args=(num_gpus, args), nprocs=num_gpus, join=True)
    else:
        print("只有一个GPU可用，使用单GPU模式")
        # 回退到单GPU模式
        import eval_optimized
        eval_optimized.eval(args)

if __name__ == '__main__':
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    main()
