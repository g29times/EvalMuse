import json
import random
import argparse
import os

def load_data(file_path):
    """加载JSON数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_data(data, file_path):
    """保存JSON数据"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def split_dataset(data_file, train_ratio=0.8, seed=42):
    """将数据集分割为训练集和验证集"""
    # 加载数据
    data = load_data(data_file)
    
    # 设置随机种子以确保可重复性
    random.seed(seed)
    
    # 随机打乱数据
    random.shuffle(data)
    
    # 计算分割点
    split_idx = int(len(data) * train_ratio)
    
    # 分割数据
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    return train_data, val_data

def main():
    # 设置控制台输出编码
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    parser = argparse.ArgumentParser(description='将数据集分割为训练集和验证集')
    parser.add_argument('--data_file', type=str, default='dataset/train_mask.json', help='数据集文件路径')
    parser.add_argument('--train_ratio', type=float, default=0.95, help='训练集占比，默认0.8')
    parser.add_argument('--seed', type=int, default=42, help='随机种子，默认42')
    parser.add_argument('--output_dir', type=str, default='dataset', help='输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 定义输出文件路径
    train_file = os.path.join(args.output_dir, 'train_split_95.json')
    val_file = os.path.join(args.output_dir, 'val_split_05.json')
    
    # 分割数据集
    train_data, val_data = split_dataset(args.data_file, args.train_ratio, args.seed)
    
    # 保存分割后的数据
    save_data(train_data, train_file)
    save_data(val_data, val_file)
    
    print(f"数据集已分割完成：")
    print(f"  - 训练集：{len(train_data)} 个样本，保存到 {train_file}")
    print(f"  - 验证集：{len(val_data)} 个样本，保存到 {val_file}")

if __name__ == '__main__':
    main()
