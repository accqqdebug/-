# TaskPrompter 模型代码包

仅包含模型源码、依赖版本和代码说明。快照来自当前服务器，保留原文件头部许可与署名。没有附带训练、优化器、评估、测试、队列、实验配置、数据集、检查点或编译二进制。

## 模型结构

图像 → ViT/Swin 任务提示骨干 → 任务特征 → 可选任务门控/SNN适配器 → 各任务预测头。

`TaskPrompterWrapper`负责骨干与预测头的组合、请求任务执行、任务特征门控及显式状态接口。空间/通道提示与多层融合在骨干中实现。

| 文件 | 职责 |
|---|---|
| models/taskprompter_wrapper.py | 多任务封装；共享/独立脉冲适配器；任务控制及状态传递 |
| models/transformers/taskprompter.py | ViT-B/ViT-L骨干、任务提示注意力、多层任务特征融合、ConvHead/DEConvHead |
| models/transformers/taskprompter_swin.py | Swin-B骨干、窗口注意力及任务提示特征提取 |
| models/spiking/task_routed_memory.py | 任务路由脉冲记忆与显式状态处理 |
| models/spiking/prompt_dynamics.py | 任务关系条件化的脉冲提示动力学 |
| intent_neuron_candidate.py | 有效阈值有界乘性调制，继承原适配器 |
| intent_analog_control.py | 任务条件无记忆连续激活对照，属于纯模型实现 |
| intent_evidence_adapter.py | 锚点控制、多尺度证据、时变阈值、事件/膜读出及可控膜通路；目标状态投影与星型特征交互 |
| detection_toolbox/det_head.py | 可选FCOS3DHead网络定义 |
| detection_toolbox/fpn.py | 可选FPN网络定义 |
| detection_toolbox/mm_builder.py | 可选3D分支的组件注册与构建依赖 |

## 核心模型与新增模型的关系

原工程的`models/`按当前源码原样保留，包括已有未提交模型改动；本包不是将新模块合并进原工程默认行为后的版本。

`IntentNeuronCandidate`继承`TaskModulatedSpikingAdapter`，`IntentAnalogControl`和`EvidenceAdapter`进一步提供不同计算路径。它们可以直接导入和构造；没有附带实验runner，也没有默认替换原Wrapper中的适配器。

`TargetStateProjection`与`execute_star_features`是可调用的状态交互原语，调用者仍需组合预测头并管理参数注册。星型交互使用两遍计算，不能视为零额外成本。

新增模块是研究候选，不代表已证明优于ANN或达到SOTA。模型参数`membrane_floor`、evidence模式和readout模式等需要随构造配置保存，仅靠state_dict不能恢复所有行为开关。

## 使用范围与依赖

核心四任务代码可以在包根目录导入：

```python
from models.taskprompter_wrapper import TaskPrompterWrapper, TaskModulatedSpikingAdapter
from models.transformers.taskprompter import taskprompter_vit_base_patch16_384, ConvHead
from intent_evidence_adapter import EvidenceAdapter, TargetStateProjection, execute_star_features
```

推荐使用原验证环境Python3.8、requirements-model.txt所列版本。Torch/CUDA安装应匹配目标机器，版本清单不是环境整包。timm较新版本可能不兼容这里的旧API。

构造完整模型仍需任务定义、输出通道、输入尺寸与骨干参数。迁移训练权重还需对应的构造开关；本包没有包含数据路径、训练配置或权重。`pretrained=False`可避免骨干构造时下载预训练参数。

3D目录只保留网络头与FPN，不包含bbox后处理、NMS CUDA扩展、检测损失或Cityscapes数据读取；不宣称可直接运行完整3D检测流程。核心四任务导入不需要MMCV。

## 校验与来源

SOURCE_MANIFEST.json记录每个复制源码的原路径、SHA256及顶层类，便于核查当前版本。verification.json记录本包在隔离目录中完成的导入、语法与小型模型前向检查；验证脚本本身没有放进代码包。

保留各文件原有许可证。原TaskPrompter文件声明CC BY-NC 4.0，可选检测组件含OpenMMLab来源；新增代码没有在此另行授予统一许可证。
