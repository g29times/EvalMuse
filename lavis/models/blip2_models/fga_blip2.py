"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lavis.common.registry import registry
from lavis.models.blip2_models.blip2_qformer import Blip2Qformer
from lavis.models.blip_models.blip_outputs import BlipOutput

class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # initial MLP param
        for name, param in self.layers.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=1.0/(self.input_size+1))
            if 'bias' in name:
                nn.init.constant_(param, val=0)
        
    def forward(self, input):
        return torch.sigmoid(self.layers(input))

@registry.register_model("fga_blip2")
class FGA_Blip2(Blip2Qformer):
    """
    BLIP Image-Text Matching (ITM) model.
    Supported model types:
        - pretrained: pretrained model
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_image_text_matching", "pretrained")
        >>> model = load_model("blip2_image_text_matching", "coco")
    """

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
    ):
        super().__init__(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            embed_dim=embed_dim,
            max_txt_len=max_txt_len,
        )
        # self.mask_proj = torch.nn.Linear(self.Qformer.config.hidden_size, 1)
        # self.weight_proj = MLP(self.Qformer.config.hidden_size)
        self.mask_proj = MLP(self.Qformer.config.hidden_size)
        # for name, parms in self.named_parameters():
        #     if '_proj' not in name:
        #         parms.requires_grad_(False)
    
    def extract_element_type(self, element_name):
        """从元素名称中提取类型"""
        if "(" in element_name and ")" in element_name:
            return element_name.split("(")[1].split(")")[0]
        return "unknown"
    
    def calculate_element_weights(self, element_names, device):
        """根据元素类型分配权重"""
        weights = torch.ones(len(element_names), dtype=torch.float).to(device)
        
        for i, name in enumerate(element_names):
            element_type = self.extract_element_type(name)
            if element_type in ["counting", "number"]:
                weights[i] = 1.5  # 增加计数类元素的权重
            elif element_type == "activity":
                weights[i] = 1.3  # 增加活动类元素的权重
            elif element_type in ["spatial", "position", "relation"]:
                weights[i] = 1.3  # 增加空间关系元素的权重
            elif element_type == "attribute":
                weights[i] = 0.9  # 略微降低属性类元素的权重（因为数量多）
            elif element_type == "object":
                weights[i] = 0.9  # 略微降低物体类元素的权重（因为数量多）
        
        return weights
    
    def get_element_thresholds(self, element_names, device):
        """为不同元素类型设置不同的阈值"""
        thresholds = torch.ones(len(element_names), dtype=torch.float).to(device) * 0.5  # 默认阈值0.5
        
        for i, name in enumerate(element_names):
            element_type = self.extract_element_type(name)
            if element_type in ["counting", "number"]:
                thresholds[i] = 0.6  # 计数类元素需要更高的阈值
            elif element_type == "activity":
                thresholds[i] = 0.55  # 活动类元素需要稍高的阈值
            elif element_type in ["spatial", "position", "relation"]:
                thresholds[i] = 0.55  # 空间关系需要稍高的阈值
        
        return thresholds

    def element_score(self, image, caption):
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            query_feat = F.normalize(
                self.vision_proj(query_output.last_hidden_state), dim=-1
            )

            # 处理文本输入 - 修复错误
            text = self.tokenizer(
                caption,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)

            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_feat = F.normalize(
                self.text_proj(text_output.last_hidden_state), dim=-1
            )

            image_feat_all = query_feat
            text_feat_all = text_feat

            mask = self.mask_proj(text_output.last_hidden_state).squeeze(dim=2)
            mask = torch.sigmoid(mask)
            
            # 使用默认阈值0.5
            mask_pred = (mask > 0.5).float()
            
            itm_score, itm_scores = self.compute_itm(
                image_feat_all, text_feat_all, query_tokens.size(1), mask_pred
            )

            return itm_score

    def compute_itm(self, image_feat, text_feat, query_token_len, mask):
        itm_scores = torch.bmm(image_feat, text_feat.transpose(1, 2))
        
        # 检查维度是否合适进行分割
        if itm_scores.size(1) > query_token_len and query_token_len > 0:
            itm_scores = itm_scores[:, query_token_len:, :] - itm_scores[:, :query_token_len, :]
            itm_scores = itm_scores[:, :, 0]
        else:
            # 如果维度不合适，直接使用原始分数
            itm_scores = itm_scores[:, :, 0]
        
        itm_score = itm_scores.mean(dim=1) * 4 + 1

        return itm_score, itm_scores

    def forward(self, samples, match_head="itm", inference = False):
        image = samples["image"]
        caption = samples["text_input"]
        
        if match_head == "itm":
            score = samples["score"]
            mask_gt = None
            token_score = None
            var = None
            split_confidence = 0.0
            attribute_confidence = 1.0
            prompt_meaningless = 0.0
            
            if 'mask' in samples:
                mask_gt = torch.tensor(samples['mask']).to(image.device).float()
                token_score = torch.tensor(samples['token_score']).to(image.device).float()
            if 'var' in samples:
                var = torch.tensor(samples['var']).to(image.device).float()
            else:
                var = torch.ones(image.size(0)).to(image.device).float()
            
            if 'split_confidence' in samples:
                split_confidence = torch.tensor(samples['split_confidence']).to(image.device).float()
            if 'attribute_confidence' in samples:
                attribute_confidence = torch.tensor(samples['attribute_confidence']).to(image.device).float()
            if 'prompt_meaningless' in samples:
                prompt_meaningless = torch.tensor(samples['prompt_meaningless']).to(image.device).float()
            
            # 提取元素名称和得分（如果有）
            element_names = []
            if 'element_score' in samples and samples['element_score'] is not None:
                for key in samples['element_score']:
                    element_names.append(key)
            
            with self.maybe_autocast():
                image_embeds = self.ln_vision(self.visual_encoder(image))
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                    image.device
                )

                query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

                query_feat = F.normalize(
                    self.vision_proj(query_output.last_hidden_state), dim=-1
                )

                # 处理文本输入 - 修复错误
                text = self.tokenizer(
                    caption,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(image.device)

                text_output = self.Qformer.bert(
                    text.input_ids,
                    attention_mask=text.attention_mask,
                    return_dict=True,
                )
                text_feat = F.normalize(
                    self.text_proj(text_output.last_hidden_state), dim=-1
                )

                image_feat_all = query_feat
                text_feat_all = text_feat

                mask = self.mask_proj(text_output.last_hidden_state).squeeze(dim=2)
                mask = torch.sigmoid(mask)
                
                itm_score, itm_scores = self.compute_itm(
                    image_feat_all, text_feat_all, query_tokens.size(1), mask
                )
            
            if inference:
                return itm_score
                
            l1_loss = torch.nn.L1Loss(reduction='mean')
            diff_score = torch.abs(itm_score - score)
            
            # 计算元素类型权重
            if element_names:
                element_weights = self.calculate_element_weights(element_names, image.device)
                # 应用元素权重到token级别的损失计算中
                diff_token_score = torch.abs(itm_scores * mask_gt - token_score)
                
                # 创建一个权重矩阵，将元素权重扩展到batch维度
                batch_size = diff_token_score.shape[0]
                token_weights = torch.ones_like(diff_token_score)
                
                # 对于每个样本，应用相应的元素权重
                for i in range(batch_size):
                    # 只对有mask的token应用权重（mask_gt > 0的位置）
                    token_weights[i] = torch.where(mask_gt[i] > 0, 
                                                 element_weights.unsqueeze(0).expand(diff_token_score.shape[1], -1)[0], 
                                                 torch.ones_like(diff_token_score[i]))
                
                # 应用权重并计算平均损失
                weighted_diff_token = (diff_token_score * token_weights).sum(dim=1) / (token_weights * mask_gt).sum(dim=1).clamp(min=1.0)
            else:
                # 如果没有元素信息，使用原始计算方式
                weighted_diff_token = torch.abs(itm_scores * mask_gt - token_score).mean(dim=1)
            
            diff_mask = torch.abs(mask - mask_gt).mean(dim=1)
            
            # 计算置信度权重
            confidence_weight = 0.5 + 0.5 * ((1 - split_confidence) * attribute_confidence * (1 - prompt_meaningless))
            
            # 调整损失函数权重比例：降低整体分数权重，提高元素级评估权重
            loss_itm = torch.mean(var * confidence_weight * (0.2 * diff_score + 0.5 * weighted_diff_token + 0.3 * diff_mask))
            
            # 添加L2正则化
            l2_reg = 0.0
            for param in self.mask_proj.parameters():
                l2_reg += torch.norm(param)
            loss_itm = loss_itm + 1e-5 * l2_reg
            
            return BlipOutput(loss=loss_itm, loss_itm=loss_itm)

        elif match_head == "itc":
            with self.maybe_autocast():
                image_embeds = self.ln_vision(self.visual_encoder(image))
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                    image.device
                )
                
                query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
                image_feats = F.normalize(
                    self.vision_proj(query_output.last_hidden_state), dim=-1
                )

                # 处理文本输入 - 修复错误
                text = self.tokenizer(
                    caption,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(image.device)

                text_output = self.Qformer.bert(
                    text.input_ids,
                    attention_mask=text.attention_mask,
                    return_dict=True,
                )
                
                text_feat = F.normalize(
                    self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
                )

                sims = torch.bmm(image_feats, text_feat.unsqueeze(-1))
                sim, _ = torch.max(sims, dim=1)
                
                itc_scores = sim * 5
                if inference:
                    return itc_scores.squeeze()
                    
                score = samples["score"]
                loss_itc = (itc_scores - score) * (itc_scores - score)
                loss_itc = loss_itc.mean()
                
                return BlipOutput(loss=loss_itc, loss_itc=loss_itc)
