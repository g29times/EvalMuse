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

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side='right')
tokenizer.add_special_tokens({"bos_token": "[DEC]"})

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

def get_index(list1,list2):
    len_list1 = len(list1)
    len_list2 = len(list2)
    for i in range(len_list2 - len_list1 + 1):
        if list2[i:i + len_list1] == list1:
            return i
    return 0

def eval(args):
    data = load_data(args.data_file, 'json')
    model, vis_processors, text_processors = load_model_and_preprocess("fga_blip2", "coco", device=device, is_eval=True)
    model.load_checkpoint(args.model_path)
    model.eval()

    # 批处理大小
    batch_size = args.batch_size
    result_list = []
    
    # 分批处理数据
    for i in tqdm(range(0, len(data), batch_size)):
        batch_data = data[i:i+batch_size]
        batch_images = []
        batch_prompts = []
        batch_prompt_ids = []
        
        # 预处理批次数据
        for item in batch_data:
            prompt = item['prompt']
            image_path = os.path.join(args.dataset_dir, item['img_path'])
            
            image = Image.open(image_path).convert("RGB")
            image = vis_processors["eval"](image).to(device)
            batch_images.append(image)
            
            processed_prompt = text_processors["eval"](prompt)
            batch_prompts.append(processed_prompt)
            batch_prompt_ids.append(tokenizer(prompt).input_ids)
        
        # 批量推理 - 改为逐个处理样本
        torch.cuda.empty_cache()
        
        for j, item in enumerate(batch_data):
            with torch.no_grad():
                # 逐个处理每个样本
                single_image = batch_images[j].unsqueeze(0)  # 添加批次维度
                single_prompt = [batch_prompts[j]]  # 转为列表，因为模型期望列表输入
                
                alignment_score, scores = model.element_score(single_image, single_prompt)
            
            elements = item['element_score'].keys()
            prompt_ids = batch_prompt_ids[j]
            
            elements_score = dict()
            for element in elements:
                element_ = element.rpartition('(')[0]
                element_ids = tokenizer(element_).input_ids[1:-1]
                
                idx = get_index(element_ids, prompt_ids)
                if idx:
                    mask = [0] * len(prompt_ids)
                    mask[idx:idx+len(element_ids)] = [1] * len(element_ids)
                    
                    mask = torch.tensor(mask).to(device)
                    elements_score[element] = ((scores * mask).sum() / mask.sum()).item()
                else:
                    elements_score[element] = 0
                    
            item['score_result'] = alignment_score.item() if isinstance(alignment_score, torch.Tensor) else alignment_score
            item['element_result'] = elements_score
            result_list.append(item)
    
    # 保存结果
    with open(args.save_path, 'w', newline='', encoding='utf-8') as file:
        json.dump(result_list, file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default='dataset/test.json')
    parser.add_argument('--save_path', type=str, default='results/result.json')
    parser.add_argument('--model_path', type=str, default='checkpoints/fga_blip2.pth')
    parser.add_argument('--dataset_dir', type=str, default='dataset/images/')
    parser.add_argument('--batch_size', type=int, default=16, help='批处理大小，根据GPU利用率调整')
    args = parser.parse_args()
    eval(args)
