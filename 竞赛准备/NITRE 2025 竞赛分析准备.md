https://shh-han.github.io/EvalMuse-project/#guideline
https://codalab.lisn.upsaclay.fr/competitions/21220#learn_the_details-evaluation
https://huggingface.co/datasets/DY-Evalab/EvalMuse

## 具体比赛策略

### 1. 代码仓库分析

#### 1.1 数据结构
图片数据集 images.zip

数据结构和特点：
    细粒度的元素分类（object, attribute, activity等）
    每个样本都有整体分数和元素级分数
    每个样本有多个人工标注（total_score和element_score都是数组）
    total_score范围是1-5
    element_score是二值的（0或1）
    数据包含了图片来源信息（如SDXL-Turbo）
    包含了prompt的类型（real/synthetic）

1. **训练数据格式** (`train_list.json`/`train_list_split_1.json`)：
   - prompt_id：提示词唯一标识
   - prompt：文本描述
   - type：提示词类型（real/synthetic）
   - img_path：对应图片路径
   - total_score：整体对齐分数（1-5分，多个标注者）
   - element_score：元素级对齐分数（0/1，多个标注者）
   - promt_meaningless：提示词是否有意义

2. **评估数据格式** (`prompt_t2i.json`)：
   - 200个精选提示词
   - 每个提示词包含细粒度元素分解
   - 元素分类：object, attribute, activity, location, material, color, counting等

3. **测试集** (`test.json`)

#### 1.2 模型实现 (FGA-BLIP2)
1. **核心架构**：
   - 基于BLIP2的Qformer架构
   - 使用EVA-CLIP作为视觉编码器（已冻结）
   - 添加MLP网络用于mask生成

2. **评分机制 fga_blip2.py**：
   - 整体评分：
     * 通过ITM head生成匹配分数
     * 使用`* 4 + 1`缩放到1-5分范围
     * 细节：
        使用model.element_score方法获取alignment_score和token级别的scores
        alignment_score是整体的对齐分数（1-5分）
        scores是每个token的细粒度分数
   - 元素评分：
     * 定位元素在prompt中的位置
     * 使用mask机制计算局部token分数
     * 0.5作为二值化阈值
     * 细节：对每个element：
        去掉类别标签（如从"puffin (animal)"提取"puffin"）
        在原始prompt中定位这个元素
        创建一个mask，只关注元素对应的token位置
        计算该区域的平均分数作为element_score
        如果找不到元素，分数为0
   - 评估指标：SRCC、PLCC和Accuracy的加权组合
   - fga_blip2.py细节：
        基础模型：使用BLIP2的Qformer架构
        评分流程：
            图像编码：使用EVA-CLIP视觉编码器
            文本编码：使用BERT tokenizer
            交叉注意力：通过Qformer处理图像和文本的交互
            评分层：
                整体评分：通过ITM head生成（范围1-5）
                元素评分：通过MLP生成mask来关注不同文本部分
        损失函数：
        ```python
            loss_itm = torch.mean(var * (diff_score + 0.1 * diff_token_score + 0.1 * diff_mask))
        ```
        diff_score：整体评分的差异
        diff_token_score：token级别评分的差异
        diff_mask：mask预测的差异
        var：样本权重（难样本有更高权重）

3. **训练配置 fga_blip2.yaml**：
   - 学习率：1e-5（初始），1e-6（最小）
   - 预热步数：100
   - 权重衰减：0.05
   - 训练轮数epoch：4
   - 批次大小：训练14，评估16
   - 使用混合精度和分布式训练
   - fga_blip2.yaml细节：
        模型配置：
            架构：fga_blip2
            预训练：加载预训练模型
            视觉编码器：冻结（freeze_vit: True）
        数据处理：
            图像大小：364x364
            使用BLIP2的图像处理器和文本处理器
        训练超参数：
            学习率：1e-5（初始），1e-6（最小）
            预热步数：100
            权重衰减：0.05
            训练轮数：4轮
            批次大小：训练14，评估16
            使用混合精度训练（amp: True）
            使用分布式训练（distributed: True）
        优化器设置：
            调度器：linear_warmup_cosine_lr
            层级学习率衰减：0.95

4. **损失函数**：
   ```python
   loss_itm = torch.mean(var * (diff_score + 0.1 * diff_token_score + 0.1 * diff_mask))
   ```
   - diff_score：整体评分差异
   - diff_token_score：token级别评分差异
   - diff_mask：mask预测差异
   - var：样本权重（难样本加权）

#### 1.3 评估指标
1. **官方评估标准**：
   - SRCC (Spearman's Rank Correlation Coefficient)
   - PLCC (Pearson's Linear Correlation Coefficient)
   - Accuracy (元素级预测准确率)
   - 最终得分：`Final_Score = PLCC / 4 + SRCC / 4 + acc / 2`

2. **评估流程**：
   - 使用logistic回归进行分数映射
   - 二值化阈值为0.5
   - 支持批量评估

### 2. 改进方向

#### 2.1 数据层面
1. **数据分析**：
   - 分析标注一致性（通过标注方差）
   - 研究不同类型元素的分布
   - 分析real和synthetic提示词的特点

2. **数据增强**：
   - 考虑提示词的变体生成
   - 图像增强技术
   - 元素级别的数据增强

#### 2.2 模型层面
1. **架构改进**：
   - 改进mask生成机制，使用更复杂的注意力机制
   - 添加元素关系建模，考虑元素间的关系（如对象和属性的关联）
   - 引入prompt类型感知（real/synthetic）

2. **训练策略**：
   - 利用标注方差进行样本加权
   - 针对不同类型元素采用不同策略
   - 设计新的损失函数组件
   - 使用更好的数据增强方法

### 3. 实施计划
1. **第一阶段：数据准备**
   - [ ] 实现数据加载和预处理，计算标注一致性
   - [ ] 进行数据统计分析不同类型元素的分布
   - [ ] 设计并实现数据增强策略

2. **第二阶段：模型改进**
   - [ ] 修改FGA-BLIP2架构的mask机制
   - [ ] 添加元素关系建模模块
   - [ ] 设计新的损失函数

3. **第三阶段：实验和优化**
   - [ ] 实现新的训练策略，分析实验结果
   - [ ] 模型集成和后处理
   - [ ] 进行对比消融实验