# QSD-Transformer / GEAR 模型代码精简包

本文件夹对应当前GEAR分类工程：**QSD神经网络、脉冲算子、LSQ量化，以及可选冻结多视图推理封装**。不包含实验脚本、训练/评估入口、预测缓存、数据、日志、权重或论文文件；不包含另一篇MaxSpike-YOLO工程，也未打包MMDetection/MMSegmentation任务代码。

## 文件

| 文件 | 用途 |
|---|---|
| `qsd_gear/models.py` | QSD主网络、卷积块、脉冲注意力、MLP、多级spike及模型工厂 |
| `qsd_gear/quan_w.py` | LSQ卷积/线性/激活量化与正尺度迁移 |
| `qsd_gear/_quan_base_plus.py` | 量化层基类、梯度缩放与舍入基础算子 |
| `qsd_gear/loading.py` | 独立模型构建及严格检查点加载；保留旧检查点有符号步长 |
| `qsd_gear/preprocess.py` | ImageNet归一化与标准多分辨率预处理 |
| `qsd_gear/ops.py` | 概率裕度与JS分歧 |
| `qsd_gear/routing.py` | 请求内嵌套选样、融合、跳级与可选冲突规则 |
| `qsd_gear/inference.py` | 真实冻结模型推理，只对选中行执行额外视图 |
| `SOURCE_MANIFEST.json` | 来源、整理改动及SHA256 |

网络计算保持来源工程实现；整理只调整包导入、去除原文件的独立演示入口，并补充不依赖实验脚本的加载接口。

已使用服务器原检查点验证：独立导入成功，分类logits及GEAR最终概率与原实现逐值一致。验证记录在 `VALIDATION.json`，验证脚本未放入本包。

## 使用

从本文件夹导入，或将它加入Python模块搜索路径。依赖版本见 `requirements.txt`，对应现有服务器环境；其PyTorch构建使用CUDA11.8。

```python
import torch
from qsd_gear import build_model, load_frozen_checkpoint

# 构建网络；此时是随机权重。
model = build_model(num_classes=1000, time_steps=1)

# 使用你已有的检查点，本包不附带权重。
model = load_frozen_checkpoint(
    '/path/to/SD_Transformer_v2-T_1.8M_1x4_w4a1t4_74_5.pth',
    device='cuda',
)
with torch.no_grad():
    logits = model(torch.randn(1, 3, 224, 224, device='cuda'))
```

可选GEAR推理入口：

```python
from PIL import Image
from qsd_gear.inference import FrozenCascadeService

gear = FrozenCascadeService(model, shape=(26, 8, 4))
result = gear([Image.open('/path/to/image.jpg').convert('RGB')])
print(result['predictions'], result['calls'])
```

完整请求最多64图；不足64图时按比例四舍五入路由数量。默认保持margin排序、标准几何与原融合；冲突仲裁、跳级和tight/hybrid几何是可选代码路径，不表示已证实精度提升。这个精简封装包含CPU选样与传输，未实现原工程CUDA Graph服务。

默认工厂名 `spikformer_8_15M_CAFormer` 是原代码名称，实际参数量见验证记录，不能从检查点文件名推算。分类数与工厂须匹配权重。当前MultiSpike是无膜状态的多级量化算子，`T`复制静态输入；本包不将其描述为递归LIF动力学。

旧检查点的signed-alpha部署路径属于冻结推理，不用于训练权重。新正尺度检查点使用原生量化实现；不对不匹配的模型字段放宽严格加载。

原始代码来源为当前服务器QSD工程，上游项目：<https://github.com/bollossom/QSD-Transformer>。保留源码中的原有引用与声明。
