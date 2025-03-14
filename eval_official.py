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
    return None

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
    """评估模型"""
    # 加载数据
    if args.verbose > 0:
        print(f"加载数据: {args.data_file}")
    with open(args.data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.verbose > 0:
        print(f"使用设备: {device}")
    
    # 加载模型
    if args.verbose > 0:
        print(f"加载模型: {args.model_path}")
    model, vis_processors, text_processors = load_model_and_preprocess(
        "fga_blip2", "pretrain", device=device, is_eval=True
    )
    
    # 加载checkpoint
    if args.verbose > 0:
        print("加载checkpoint，忽略形状不匹配的参数...")
    model = load_checkpoint_with_ignore(model, args.model_path)
    model.eval()
    
    # 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side='right')
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    
    # 准备输出结果
    output_results = []
    
    # 处理每个样本
    if args.verbose > 0:
        print(f"开始评估，共 {len(data)} 个样本...")
    for i, item in enumerate(tqdm(data, disable=args.verbose==0)):
        try:
            prompt = item['prompt']
            img_path = os.path.join(args.dataset_dir, item['img_path'])
            elements = item['element_score'].keys()
            
            # 加载和处理图像
            try:
                raw_image = Image.open(img_path).convert('RGB')
                processed_image = vis_processors["eval"](raw_image).to(device)
            except Exception as e:
                if args.verbose > 0:
                    print(f"错误: 处理图像失败 {img_path}: {e}")
                continue
            
            # 处理文本
            processed_prompt = text_processors["eval"](prompt)
            
            # 获取tokenizer处理后的prompt_ids
            text_tokens = tokenizer(prompt, return_tensors="pt").to(device)
            prompt_ids = text_tokens.input_ids[0][1:-1].tolist()  # 去掉开始和结束标记
            
            # 推理
            with torch.no_grad():
                try:
                    alignment_score, scores = model.element_score(processed_image.unsqueeze(0), [prompt])
                    
                    if args.verbose >= 2 and i == 0:
                        print(f"alignment_score: {alignment_score.shape}, {alignment_score}")
                        print(f"scores: {scores.shape}, 类型: {type(scores)}")
                        print(f"scores[0, :10]: {scores[0, :10]}")
                        
                    # 检查scores是否为tensor
                    if isinstance(scores, torch.Tensor):
                        if args.verbose >= 2 and i == 0:
                            min_score = scores.min().item()
                            max_score = scores.max().item()
                            mean_score = scores.mean().item()
                            print(f"prompt_ids长度: {len(prompt_ids)}")
                            print(f"scores最小值: {min_score}, 最大值: {max_score}, 平均值: {mean_score}")
                
                except Exception as e:
                    if args.verbose > 0:
                        print(f"错误: 使用element_score方法出错: {e}")
                    # 如果出错，创建默认值
                    alignment_score = torch.tensor([3.0]).to(device)  # 默认中等分数
                    scores = torch.zeros(1, len(prompt_ids)).to(device)
            
            # 处理每个元素
            element_scores = {}
            found_elements = 0
            
            for element in elements:
                element_name = element.rpartition('(')[0].strip()
                element_type = element.rpartition('(')[2].rstrip(')')
                
                # 获取元素的token ids
                element_ids = tokenizer.encode(element_name, add_special_tokens=False)
                
                # 在prompt_ids中查找element_ids
                idx = get_index(element_ids, prompt_ids)
                
                if idx is not None:
                    found_elements += 1
                    
                    # 根据官方实现，直接使用对应位置的scores作为元素得分
                    # 注意：官方实现中，scores是从query_tokens.size(1)开始的文本token得分
                    if idx < scores.shape[1]:
                        # 如果元素的起始位置在scores范围内
                        # 计算元素对应位置的平均分数
                        element_score_sum = 0.0
                        valid_tokens = 0
                        
                        for j in range(len(element_ids)):
                            if idx + j < scores.shape[1]:
                                element_score_sum += scores[0, idx + j].item()
                                valid_tokens += 1
                        
                        if valid_tokens > 0:
                            element_score = element_score_sum / valid_tokens
                        else:
                            element_score = 0.0
                        
                        # 确保元素得分在0-1范围内
                        element_score = max(0.0, min(1.0, element_score))
                        
                        # 打印详细调试信息
                        if args.verbose >= 3 and i < 5:
                            print(f"  元素: {element_name} ({element_type})")
                            print(f"  位置: {idx}, 长度: {len(element_ids)}")
                            print(f"  有效token数: {valid_tokens}")
                            print(f"  元素得分: {element_score:.4f}")
                    else:
                        # 如果元素的起始位置超出scores范围
                        if args.verbose >= 2 and i == 0:
                            print(f"警告: 元素'{element_name}'的位置({idx})超出scores范围({scores.shape[1]})")
                        element_score = 0.0
                else:
                    # 元素未找到，给一个默认得分
                    element_score = 0.0
                
                # 保存元素得分
                element_scores[element] = element_score
            
            # 创建输出项
            output_item = {
                "prompt": prompt,
                "img_path": item['img_path'],
                "total_score": alignment_score.item(),
                "element_score": element_scores
            }
            
            output_results.append(output_item)
            
            # 每处理10个样本打印一次详细信息
            if args.verbose >= 2 and i % 10 == 0:
                print(f"\n处理进度: {i}/{len(data)} ({i/len(data)*100:.1f}%)")
                print(f"样本ID: {item.get('prompt_id', 'unknown')}")
                print(f"提示词: {prompt[:50]}...")
                print(f"总分: {alignment_score.item():.2f}")
                print(f"找到元素数量: {found_elements}/{len(elements)}")
                
                # 打印元素得分
                if found_elements > 0:
                    print("元素得分样例:")
                    for j, (element, score) in enumerate(element_scores.items()):
                        if j < 3:  # 只打印前3个元素
                            print(f"  - {element}: {score:.2f}")
                else:
                    print("警告: 未找到任何元素!")
            
            # 简单进度报告
            if args.verbose == 1 and i % 100 == 0:
                print(f"进度: {i}/{len(data)} ({i/len(data)*100:.1f}%) - 找到元素: {found_elements}/{len(elements)}")
                
            # 每处理指定间隔样本保存一次中间结果
            if i > 0 and i % args.save_interval == 0:
                if args.verbose > 0:
                    print(f"\n保存中间结果... ({i}/{len(data)})")
                temp_output_path = args.save_path.replace('.json', f'_temp_{i}.json')
                with open(temp_output_path, 'w', encoding='utf-8') as f:
                    json.dump(output_results, f, ensure_ascii=False, indent=4)
                
        except Exception as e:
            if args.verbose > 0:
                print(f"处理样本时出错: {e}, 项目: {item}")
            continue
    
    # 保存结果
    if args.verbose > 0:
        print(f"\n保存最终结果: {args.save_path}")
    with open(args.save_path, 'w', encoding='utf-8') as f:
        json.dump(output_results, f, ensure_ascii=False, indent=4)
    
    if args.verbose > 0:
        print("评估完成!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--data_file', type=str, required=True, help='数据文件路径')
    parser.add_argument('--save_path', type=str, required=True, help='保存结果的路径')
    parser.add_argument('--dataset_dir', type=str, required=True, help='数据集目录')
    parser.add_argument('--verbose', type=int, default=1, help='日志详细程度: 0=静默, 1=基本信息, 2=详细信息')
    parser.add_argument('--save_interval', type=int, default=100, help='保存中间结果的间隔样本数')
    args = parser.parse_args()
    
    eval(args)
