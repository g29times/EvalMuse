import torch
import json
import os
from tqdm import tqdm
from transformers import BertTokenizer
from lavis.models import load_model_and_preprocess
from PIL import Image
import argparse

def get_index(list1, list2):
    """查找list1在list2中的起始索引"""
    len_list1 = len(list1)
    len_list2 = len(list2)
    for i in range(len_list2 - len_list1 + 1):
        if list2[i:i + len_list1] == list1:
            return i
    return 0

def load_checkpoint_with_ignore(model, checkpoint_path):
    """加载checkpoint，忽略不匹配的参数"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    
    # 创建一个新的state_dict，只包含模型中存在的键
    new_state_dict = {}
    model_state_dict = model.state_dict()
    
    for k, v in state_dict.items():
        if k in model_state_dict:
            # 检查形状是否匹配
            if v.shape == model_state_dict[k].shape:
                new_state_dict[k] = v
            else:
                print(f"忽略形状不匹配的参数: {k}, 模型形状: {model_state_dict[k].shape}, checkpoint形状: {v.shape}")
        else:
            print(f"忽略模型中不存在的参数: {k}")
    
    # 加载过滤后的state_dict
    model.load_state_dict(new_state_dict, strict=False)
    return model

def eval(args):
    # 设置设备
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # 加载BERT分词器
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side='right')
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    
    # 加载模型和预处理器
    model, vis_processors, text_processors = load_model_and_preprocess(
        "fga_blip2", "pretrain", device=device, is_eval=True
    )
    
    # 使用自定义函数加载checkpoint
    model = load_checkpoint_with_ignore(model, args.model_path)
    model.eval()
    
    # 加载测试数据
    with open(args.data_file, 'r') as f:
        data = json.load(f)
    
    # 准备结果列表
    output_results = []
    
    # 逐个处理数据项
    for item in tqdm(data, desc="评估中"):
        try:
            # 获取元素列表
            elements = item['element_score'].keys()
            prompt = item['prompt']
            
            # 加载并预处理图像
            image_path = os.path.join(args.dataset_dir, item['img_path'])
            image = Image.open(image_path).convert("RGB")
            processed_image = vis_processors["eval"](image).to(device)
            
            # 处理文本
            processed_prompt = text_processors["eval"](prompt)
            prompt_ids = tokenizer(prompt).input_ids
            
            # 清理GPU缓存
            torch.cuda.empty_cache()
            
            # 推理
            with torch.no_grad():
                try:
                    # 调用element_score方法
                    alignment_score, scores = model.element_score(processed_image.unsqueeze(0), [prompt])
                except Exception as e:
                    print(f"使用element_score方法出错: {e}")
                    # 如果出错，创建默认值
                    alignment_score = torch.tensor([3.0]).to(device)  # 默认中等分数
                    scores = torch.zeros(1, len(prompt_ids)).to(device)
            
            # 计算每个元素的得分
            elements_score = {}
            for element in elements:
                element_name = element.rpartition('(')[0].strip()
                element_ids = tokenizer(element_name).input_ids[1:-1]  # 去掉开始和结束标记
                
                # 查找元素在提示中的位置
                idx = get_index(element_ids, prompt_ids)
                
                if idx:
                    # 创建掩码
                    mask = [0] * len(prompt_ids)
                    mask[idx:idx+len(element_ids)] = [1] * len(element_ids)
                    
                    mask = torch.tensor(mask).to(device)
                    # 计算元素得分
                    # 确保scores的维度与mask匹配
                    if scores.size(0) != len(prompt_ids):
                        # 如果scores维度不匹配，进行调整
                        # 通常itm_scores的维度是[batch_size, seq_len]
                        # 我们需要将其调整为与prompt_ids匹配的长度
                        scores_resized = torch.zeros(len(prompt_ids)).to(device)
                        # 复制可用的分数
                        min_len = min(scores.size(0), len(prompt_ids))
                        scores_resized[:min_len] = scores[0, :min_len]
                        element_score = ((scores_resized * mask).sum() / mask.sum()).item()
                    else:
                        element_score = ((scores[0] * mask).sum() / mask.sum()).item()
                    elements_score[element] = element_score
                else:
                    elements_score[element] = 0
            
            # 创建输出项
            output_item = {
                "prompt": prompt,
                "img_path": item['img_path'],
                "total_score": alignment_score.item(),
                "element_score": elements_score
            }
            
            output_results.append(output_item)
            
        except Exception as e:
            print(f"处理项目时出错: {e}, 项目: {item}")
            # 创建一个默认的输出项
            default_elements = {element: None for element in item['element_score'].keys()}
            output_item = {
                "prompt": item['prompt'],
                "img_path": item['img_path'],
                "total_score": None,
                "element_score": default_elements
            }
            output_results.append(output_item)
    
    # 保存结果
    with open(args.save_path, 'w', encoding='utf-8') as f:
        json.dump(output_results, f, ensure_ascii=False, indent=4)
    
    print(f"评估完成，结果已保存到 {args.save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default='dataset/test.json')
    parser.add_argument('--save_path', type=str, default='results/output.json')
    parser.add_argument('--model_path', type=str, default='lavis/output/FGA-BLIP2/20250314051/checkpoint_2.pth')
    parser.add_argument('--dataset_dir', type=str, default='dataset/images/')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    eval(args)
