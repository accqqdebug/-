# 模型代码梳理

## 1. 主干数据流

`RGB图像 [B,3,H,W] → 复制时间维 [T,B,3,H,W] → 卷积编码/降采样 → block3 → downsample4 → block4 → 全局空间平均 → Multispike → 量化分类头 → 时间平均 → logits [B,K]`。

默认 `spikformer_8_15M_CAFormer` 使用嵌入宽度 `[64,128,256,360]`、8个attention heads。`depths=8` 分为block3的6个MS_Block及block4的2个MS_Block。源码也保留256/384/512/768宽度和12层等工厂定义。

默认1000类模型实测参数量为 **15,245,862**。已在CPU FP32上验证64与224两种输入尺寸的logits和原工程逐值一致，GEAR最终概率也逐值一致；详细记录见 `VALIDATION.json`。

## 2. 模块对应

| 模块 | 代码位置 | 作用 |
|---|---|---|
| `MS_DownSampling` | models.py | 脉冲预激活、量化卷积、BN与空间尺寸变换 |
| `MS_ConvBlock` | models.py | 前端卷积特征提取 |
| `MS_Attention_RepConv_qkv_id` | models.py | 生成量化/脉冲Q/K/V，使用`Q(KᵀV)`关联与投影 |
| `MS_MLP` | models.py | 脉冲与卷积实现的通道变换 |
| `MS_Block` | models.py | attention/MLP残差、layer scale与drop path |
| `Multispike` | models.py | `floor(clamp(4x,0,4)+0.5)/4` |
| `Multispike_att` | models.py | 同整数等级但输出除以2；统计尺度与普通spike不同 |
| `Conv2dLSQ/Conv1dLSQ/LinearLSQ` | quan_w.py | 权重量化与可选激活量化 |
| `_PositiveLSQScaleMixin` | quan_w.py | 正量化步长及旧signed-alpha迁移 |
| `TemporalBatchDropPath` | models.py | 训练时按B采样、沿T共享的残差丢弃 |

## 3. 检查点加载

原网络量化层支持正尺度迁移，但旧发布权重的部署基线需要原始有符号量化步长。`loading.py` 在旧权重模式下先保存原alpha，严格加载其他字段，再把原步长注册为buffer，用原网格计算有效权重。这样无需依赖原工程的实验加载脚本。

只允许补齐未启用的嵌套激活字段与量化迁移标记；发布权重中明确未使用的 `top_conv.topconv.weight` 被单独排除。其他缺失或多余参数仍报错。

## 4. GEAR在网络之外的作用

GEAR共用同一个冻结QSD模型。先对整个请求计算224廉价视图，再选26/64执行288视图；在子集中重排后选8/64执行flip动作，再选4/64执行320精炼。

标准概率融合：

- 第一阶段：`P1 = 0.3*P0 + 0.7*P288`。
- 第二阶段：`P2 = 0.3*P1 + 0.7*(0.2*P0 + 0.8*Pflip)`。
- 第三阶段：`P3 = 0.7*P2 + 0.3*P320`。

本包flip动作与当前缓存生成器一致：对原288与水平翻转288的logits先平均，再softmax；原288 logits直接复用。它不是又一个独立训练模型。

这个包保留模型与推理逻辑；探针校准、数据集评估、实验搜索、能耗测量和统计分析均未包含。
