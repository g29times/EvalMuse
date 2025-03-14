"""
将我们的结果文件转换为官方提交格式
"""

import json
import argparse

def convert_to_submission_format(input_file, output_file):
    """将我们的结果文件转换为官方提交格式"""
    with open(input_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    submission = []
    for item in results:
        submission_item = {
            "prompt": item["prompt"],
            "img_path": item["img_path"],
            "total_score": item["score_result"],
            "element_score": {}
        }
        # 转换元素预测结果
        for element, score in item["element_result"].items():
            submission_item["element_score"][element] = score
        
        submission.append(submission_item)
    
    print(f"转换完成：{len(submission)}个样本")
    print(f"第一个样本示例：{submission[0]}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(submission, f, indent=4, ensure_ascii=False)
    
    print(f"结果已保存到：{output_file}")

def convert_to_test_format(input_file, output_file):
    """将我们的结果文件转换为官方测试格式（total_score和element_score为null）"""
    with open(input_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    submission = []
    for item in results:
        submission_item = {
            "prompt": item["prompt"],
            "img_path": item["img_path"],
            "total_score": None,
            "element_score": {}
        }
        # 转换元素预测结果，设置为null
        for element in item["element_result"].keys():
            submission_item["element_score"][element] = None
        
        submission.append(submission_item)
    
    print(f"转换完成：{len(submission)}个样本")
    print(f"第一个样本示例：{submission[0]}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(submission, f, indent=4, ensure_ascii=False)
    
    print(f"结果已保存到：{output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="转换结果文件格式")
    parser.add_argument("--input", type=str, required=True, help="输入文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出文件路径")
    parser.add_argument("--mode", type=str, choices=["submit", "test"], default="submit", 
                        help="转换模式：submit-提交格式，test-测试格式（值为null）")
    
    args = parser.parse_args()
    
    if args.mode == "submit":
        convert_to_submission_format(args.input, args.output)
    else:
        convert_to_test_format(args.input, args.output)
