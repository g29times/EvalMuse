import json
import numpy as np
from scipy.stats import pearsonr, spearmanr
import argparse
from utils import load_data
from scipy.optimize import curve_fit

def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    """逻辑函数，用于非线性映射预测分数"""
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat

def fit_function(y_label, y_output):
    """使用逻辑函数拟合预测分数"""
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)
    return y_output_logistic

def binarize(lst):
    """将列表中的元素按 0.5 阈值进行二值化"""
    return [1 if x >= 0.5 else 0 for x in lst]

def calculate_accuracy(true_list, pred_list):
    """计算二值化后的准确率"""
    true_bin = binarize(true_list)
    pred_bin = binarize(pred_list)
    
    # 计算相同元素的数量
    correct = sum([1 for t, p in zip(true_bin, pred_bin) if t == p])
    # 计算准确率
    accuracy = correct / len(true_list)
    return accuracy

def calculate_metrics(result_file, ground_truth_file=None):
    """计算评估指标"""
    # 加载预测结果
    results = load_data(result_file, 'json')
    
    # 如果提供了真实标签文件，则加载
    if ground_truth_file:
        ground_truth = load_data(ground_truth_file, 'json')
        # 创建ID到数据的映射
        gt_map = {item['prompt_id']: item for item in ground_truth}
    else:
        gt_map = None
    
    # 收集总分和元素分数
    total_scores_pred = []
    total_scores_true = []
    
    all_element_scores_pred = []
    all_element_scores_true = []
    
    matched_samples = 0
    
    # 处理每个样本
    for item in results:
        if 'score_result' not in item or 'element_result' not in item:
            continue
            
        # 总分预测
        total_score_pred = item['score_result']
        
        # 元素分数预测
        element_scores = item['element_result']
        
        # 如果有真实标签
        if gt_map:
            gt_item = gt_map.get(item['prompt_id'])
            if gt_item:
                matched_samples += 1
                
                # 收集总分
                if gt_item['total_score'] is not None:
                    total_scores_pred.append(total_score_pred)
                    total_scores_true.append(gt_item['total_score'])
                
                # 收集元素分数
                for element, pred_score in element_scores.items():
                    if element in gt_item['element_score'] and gt_item['element_score'][element] is not None:
                        all_element_scores_pred.append(pred_score)
                        all_element_scores_true.append(gt_item['element_score'][element])
        else:
            # 如果没有真实标签，仍然收集预测分数用于统计
            total_scores_pred.append(total_score_pred)
            for element, score in element_scores.items():
                all_element_scores_pred.append(score)
    
    # 计算指标
    metrics = {}
    
    # 如果有真实标签，计算相关系数
    if gt_map and len(total_scores_true) > 0:
        # 计算总分的PLCC和SRCC
        try:
            # 使用逻辑函数拟合后计算PLCC
            y_output_logistic = fit_function(total_scores_true, total_scores_pred)
            plcc = pearsonr(y_output_logistic, total_scores_true)[0]
            
            # 计算SRCC
            srcc = spearmanr(total_scores_pred, total_scores_true)[0]
            
            # 计算元素分数的准确率
            if len(all_element_scores_true) > 0:
                acc = calculate_accuracy(all_element_scores_true, all_element_scores_pred)
            else:
                acc = None
                
            # 计算最终得分
            if acc is not None:
                final_score = plcc/4 + srcc/4 + acc/2
            else:
                final_score = (plcc + srcc) / 2
                
            metrics = {
                'PLCC': plcc,
                'SRCC': srcc,
                'Accuracy': acc,
                'Final Score': final_score,
                'Matched Samples': matched_samples
            }
        except Exception as e:
            print(f"计算相关系数时出错: {e}")
    
    # 统计预测分数的分布
    pred_stats = {}
    if len(total_scores_pred) > 0:
        pred_stats.update({
            'Total Score Mean': np.mean(total_scores_pred),
            'Total Score Std': np.std(total_scores_pred),
            'Total Score Min': min(total_scores_pred),
            'Total Score Max': max(total_scores_pred),
        })
    
    if len(all_element_scores_pred) > 0:
        pred_stats.update({
            'Element Score Mean': np.mean(all_element_scores_pred),
            'Element Score Std': np.std(all_element_scores_pred),
            'Element Score Min': min(all_element_scores_pred),
            'Element Score Max': max(all_element_scores_pred)
        })
    
    return metrics, pred_stats

if __name__ == '__main__':
    # 设置控制台输出编码
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    parser = argparse.ArgumentParser(description='Calculate evaluation metrics')
    parser.add_argument('--result_file', type=str, required=True, help='Path to result JSON file')
    parser.add_argument('--ground_truth', type=str, help='Path to ground truth JSON file (optional)')
    parser.add_argument('--output', type=str, help='Path to output metrics JSON file (optional)')
    parser.add_argument('--debug', action='store_true', help='Print debug information')
    
    args = parser.parse_args()
    
    # 打印文件信息
    print(f"结果文件: {args.result_file}")
    print(f"真实标签文件: {args.ground_truth if args.ground_truth else '未提供'}")
    
    # 加载数据
    try:
        results = load_data(args.result_file, 'json')
        print(f"成功加载结果文件，包含 {len(results)} 个样本")
        
        if args.ground_truth:
            ground_truth = load_data(args.ground_truth, 'json')
            print(f"成功加载真实标签文件，包含 {len(ground_truth)} 个样本")
            
            # 检查数据格式
            if args.debug and len(ground_truth) > 0:
                sample = ground_truth[0]
                print("\n真实标签样本格式:")
                print(f"  prompt_id: {sample.get('prompt_id')}")
                print(f"  total_score: {sample.get('total_score')}")
                print(f"  element_score keys: {list(sample.get('element_score', {}).keys())[:3]}...")
            
            if args.debug and len(results) > 0:
                sample = results[0]
                print("\n预测结果样本格式:")
                print(f"  prompt_id: {sample.get('prompt_id')}")
                print(f"  score_result: {sample.get('score_result')}")
                print(f"  element_result keys: {list(sample.get('element_result', {}).keys())[:3]}...")
    except Exception as e:
        print(f"加载数据时出错: {e}")
        sys.exit(1)
    
    # 计算评估指标
    metrics, pred_stats = calculate_metrics(args.result_file, args.ground_truth)
    
    # 打印指标
    print("\n===== 评估指标 =====")
    if metrics:
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")
    else:
        print("未能计算评估指标，可能是因为没有提供真实标签或标签格式不匹配")
    
    print("\n===== 预测分数统计 =====")
    for stat, value in pred_stats.items():
        print(f"{stat}: {value:.4f}")
    
    # 保存指标到文件
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump({
                'metrics': metrics,
                'prediction_stats': pred_stats
            }, f, indent=4, ensure_ascii=False)
        print(f"\n指标已保存到 {args.output}")
