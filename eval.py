import torch
import json
import csv
from transformers import BertTokenizer
from tqdm import tqdm
from lavis.models import load_model_and_preprocess, load_model
import os
from PIL import Image
from utils import compute_metrics
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side='right')
tokenizer.add_special_tokens({"bos_token": "[DEC]"})

def binarize(lst):
    # 将列表中的元素按 0.5 阈值进行二值化
    return [1 if x >= 0.5 else 0 for x in lst]

def calculate_accuracy(true_list, pred_list):
    # 先二值化两个列表
    # breakpoint()
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

def load_csv_as_dict_list(file_path):
    dict_list = []
    with open(file_path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            dict_list.append(row)
    return dict_list

def load_json_as_dict_list(file_path):
    with open(file_path, mode='r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    return data

def load_data(file_path, file_type):
    if file_type == 'csv':
        return load_csv_as_dict_list(file_path)
    elif file_type == 'json':
        return load_json_as_dict_list(file_path)
    else:
        raise ValueError("Unsupported file type. Please use 'csv' or 'json'.")
    

def get_index(list1,list2):
    len_list1 = len(list1)
    len_list2 = len(list2)
    for i in range(len_list2 - len_list1 + 1):
        if list2[i:i + len_list1] == list1:
            return i
    return 0

def is_sublist(lst1, lst2):
    return str(lst1)[1:-1] in str(lst2)[1:-1]
# max_len = 0
# for item in tqdm(data):
#     prompt = item['prompt']
#     ids = tokenizer(prompt).input_ids
#     if max_len < len(ids):
#         max_len = len(ids)
#     breakpoint()
# print(max_len)

data = load_data('dataset/data_test_new.json','json')
error = 0
data_new = []
len_flow = 0

model, vis_processors, text_processors = load_model_and_preprocess("fga_blip2", "coco", device=device, is_eval=True)
# model = load_model("blip2_alignment", "coco", device=device, is_eval=True, checkpoint='/mnt/bn/hanshuhao1/mlx/users/hanshuhao/LAVIS/lavis/output/BLIP2/Alignment_ft/20240829162/checkpoint_49.pth')
model.load_checkpoint("lavis/output/BLIP2/Alignment_ft_mask/blip2_var_mask_split/20241203120/checkpoint_3.pth")
model.eval()
element_score_gt = []
element_score = []
alignment_score_gt_list = []
alignment_score_list = []

data_blip2_element = []
for item in tqdm(data):
    elements = item['element_score'].keys()
    prompt = item['prompt']
    
    image = os.path.join('../T2IEVAL-40k/dataset', item['img_path'])

    image = Image.open(image).convert("RGB")
    image = vis_processors["eval"](image).to(device)
    prompt = text_processors["eval"](prompt)
    prompt_ids = tokenizer(prompt).input_ids
    # breakpoint()

    torch.cuda.empty_cache()
    with torch.no_grad():
        alignment_score, scores = model.element_score(image.unsqueeze(0),[prompt])

    blip2_elements = dict()
    for element in elements:
        element_ = element.rpartition('(')[0]
        element_ids = tokenizer(element_).input_ids[1:-1]
        # breakpoint()

        idx = get_index(element_ids,prompt_ids)
        # breakpoint()
        if idx:
            mask = [0] * len(prompt_ids)
            mask[idx:idx+len(element_ids)] = [1] * len(element_ids)
            
            mask = torch.tensor(mask).to(device)
            element_score_gt.append(item['element_score'][element])
            element_score.append(((scores * mask).sum() / mask.sum()).item())
            blip2_elements[element] = ((scores * mask).sum() / mask.sum()).item()
        else:
            blip2_elements[element] = 0
    item['ele_blip2'] = blip2_elements
    item['fga_blip2'] = alignment_score.item()
    alignment_score_gt_list.append(item['total_score'])
    alignment_score_list.append(alignment_score.item())
    data_blip2_element.append(item)
acc,loss =  calculate_accuracy(element_score_gt,element_score)     

print(acc)
print(loss)

SRCC_blip, KRCC_blip, PLCC_blip, RMSE_blip = compute_metrics(list(alignment_score_gt_list), list(alignment_score_list))
print(f"SRCC_blip_itm: {SRCC_blip:.4f}, KRCC_blip_itm: {KRCC_blip:.4f}, PLCC_blip_itm: {PLCC_blip:.4f}, RMSE_blip_itm: {RMSE_blip:.4f}")

save_file = 'results/result_fga_blip2.json'
# 打开json文件进行写入
with open(save_file, 'w', newline='', encoding='utf-8') as file:
    json.dump(data_blip2_element, file, ensure_ascii=False, indent=4)


