import argparse
import json
import os
import torch
from PIL import Image
from tqdm import tqdm
from lavis.models import load_model_and_preprocess
from lavis.processors.blip_processors import BlipCaptionProcessor
import torch.nn.functional as F

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
    # 加载模型和预处理
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name="fga_blip2", model_type="pretrain", is_eval=True, device=device
    )
    
    # 使用自定义函数加载checkpoint，忽略不匹配的参数
    model = load_checkpoint_with_ignore(model, args.model_path)
    model.eval()
    
    # 加载测试数据
    with open(args.data_file, 'r') as f:
        data = json.load(f)
    
    results = []
    batch_size = args.batch_size
    
    # 批处理评估
    for i in tqdm(range(0, len(data), batch_size), desc="评估中"):
        batch_data = data[i:i+batch_size]
        batch_images = []
        batch_captions = []
        
        for item in batch_data:
            # 根据数据格式调整图像路径获取方式
            if 'img_path' in item:
                image_path = os.path.join(args.dataset_dir, item['img_path'])
            else:
                print(f"警告: 数据项缺少img_path字段: {item}")
                continue
                
            image = Image.open(image_path).convert('RGB')
            processed_image = vis_processors["eval"](image).to(device)
            batch_images.append(processed_image)
            batch_captions.append(item['prompt'])
        
        if not batch_images:
            continue
            
        # 堆叠批次图像
        batch_images = torch.stack(batch_images)
        
        # 批量推理
        with torch.no_grad():
            samples = {"image": batch_images, "text_input": batch_captions}
            scores = model(samples, inference=True)
            
            # 确保scores是CPU张量并转换为Python列表
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().tolist()
        
        # 保存结果
        for j, item in enumerate(batch_data):
            if j >= len(scores):
                continue
                
            score = scores[j]
            
            # 创建与原始数据格式相同的结果项
            result_item = item.copy()
            result_item["total_score"] = score
            
            # 如果有元素得分字段，为每个元素设置得分
            if "element_score" in item and item["element_score"]:
                for element in result_item["element_score"]:
                    # 这里我们简单地将总分作为每个元素的分数
                    # 实际应用中可能需要更复杂的元素级评分逻辑
                    result_item["element_score"][element] = score
            
            results.append(result_item)
    
    # 保存结果
    with open(args.save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"评估完成，结果已保存到 {args.save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default='dataset/test.json')
    parser.add_argument('--save_path', type=str, default='results/result.json')
    parser.add_argument('--model_path', type=str, default='lavis/output/FGA-BLIP2/20250314051/checkpoint_2.pth')
    parser.add_argument('--dataset_dir', type=str, default='dataset/images/')
    parser.add_argument('--batch_size', type=int, default=32, help='批处理大小，根据GPU利用率调整')
    args = parser.parse_args()
    eval(args)
