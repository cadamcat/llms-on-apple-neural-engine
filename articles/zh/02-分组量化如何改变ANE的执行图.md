# 分组量化如何改变 ANE 的执行图：精度、兼容性与 SplitConv 成本

[English](../02-group-quantization-and-split.md) | 中文

2026-09-10 · [系列索引](README.md)

分组量化的 scale 不仅决定解码后的权重数值，也会影响图能如何表达、编译器能否接受它，以及执行时的成本。本文把一个小型兼容性探针和一个同权重性能消融放在一起，分析这几个约束之间的联系。

## 1. 权重存储只是第一个检查环节

在 LLM 推理中，分组量化常被描述为权重存储格式：每若干输入通道共享一个 scale，代码与 scale 相乘后参与矩阵乘法。但在 Apple 推理栈中，还涉及三个独立问题：资产是否正确、编译器是否接受这种 scale 粒度、运行时是否真的让算子参与 ANE。导出和 CPU 数值正确，不意味着会被 ANE 执行。

本文使用两类无模型数据：性能实验是 512 通道合成链，权重为固定种子 Hadamard 矩阵，输入为 Q8 网格向量平铺到 4096 个位置；兼容性实验是 64 输出通道、K64 的小型分组权重和单位阵输入。它们回答不同问题，均不承担 LLM 质量验收。[方法文档](../../docs/METHODS.md)规定证据边界。

## 2. K32 分组 scale 的数学含义

设一个输出通道的 64 个输入权重代码为 q，每个代码是有符号 INT4。K64 分组量化把输入通道分成两个 K32 组，各组拥有独立 scale `s₀`、`s₁`：

$$
W = \operatorname{concat}(s_0q_0, s_1q_1),\qquad
(Wx)_o = \sum_{g\in\{0,1\}} s_g\sum_{k\in g}q_{o,k}x_k .
$$

这里的 scale 是沿输入通道分组、按输出通道存储的 per-input-group 参数，不是 activation scale，也不是整个输出通道共用一个 per-cout weight scale。

这一区别正是拆分策略的关键。把 K64 写成两个 K32 卷积时，每个子卷积的每个输出通道只需要一个 weight scale：

$$
W x = (s_0q_0)x_0 + (s_1q_1)x_1.
$$

在实数精确算术下，这与原始分组权重等价，推广到 K512，按每组 K32 分解后，图也相应改变：每层变成 16 个 K32 卷积和 15 个加法；两层共 32 个卷积、30 个加法。

![同权重 K512 宽卷积与十六个 K32 部分卷积加平衡 FP16 归约的两层结构对比。](../../docs/figures/split-structure.svg)

图 1：两条路径各执行一次完整图的 host prediction。拆分图两层合计 32 个卷积、30 个加法与一次层间 QDQ；下方另外标注 K64 兼容性探针，避免与吞吐图混淆。

图中的数量是导出图内操作数，不是宿主调用次数。两个层和中间 QDQ 被导出为一个模型函数；一次预测执行整张图。

图等价不等于浮点执行等价。拆分参考在部分和、平衡树和层间 QDQ 处有 FP16 边界，宽 K 路径则不同，因此使用各自在实验前固定的数值参考。见[方法文档](../../docs/METHODS.md)。

## 3. 宽 K 正例说明了什么

在 128 层宽 K512 链上，Core AI W8A8 吞吐为 34.95–35.06 T ops/s，A8W4 为 35.07–35.09 T；同轮配对中量化路径的速度约为 FP16 的 1.86–1.87×。[新吞吐记录](../../results/fresh/throughput.json)显示计时配置通过数值与 ANE 控制。

这些测量为固定图、芯片和工具链上的低比特加速提供了正对照。低熵 ±1 权重和易表达 scale 不能外推到任意 Q4_0、LLM throughput 或能效；Core AI 无逐算子执行追踪，且 `physical_INT8_proven=false`。

## 4. SplitConv 的执行成本

对同一套合成桥接图，Core AI A8W4 宽 K512 与 K32 SplitConv 都通过了数值和 ANE 控制。每个配置各三个独立进程，每进程四项控制后执行 10 次暖机、30 次测量；三轮顺序为宽/拆、拆/宽、宽/拆。三组新配对中，宽路径的速度分别为 split 路径的 4.10×、4.05×、3.96×，范围为 3.96–4.10×。[Split 测量](../../docs/MEASUREMENTS.md)给出对应的每个配置范围：宽路径 0.432–0.446 ms，split 路径 1.764–1.772 ms。

这不是“分组量化必然慢四倍”，而是一个图结构观察。K32 拆分增加图内卷积操作、部分和与加法树，并改变融合、调度、窄 K 内核及 FP16 归约边界。现有测量没有逐项隔离这些因素，只能说完整拆分图在此主机上显著变慢。两条路径采用各自的数值参考；同权重并不代表浮点输出逐位相同。

历史材料中，宽路径的速度为 split 路径的 4.44×，每个配置各一个进程，但属于不同运行。[历史选择记录](../../results/historical/selected-evidence.json)与当前三组新进程不能合并，也不能把多次运行的差别直接解释为某项软件改动的因果效应。

![三个独立进程配对中，宽 K512 耗时约 0.43–0.45 毫秒，K32 拆分约 1.76–1.77 毫秒。](../../docs/figures/split-latency.svg)

图 2：连线连接同一轮的宽/拆分配置，不代表随时间变化。横轴从零开始，拆分 p50 除以宽 K p50 得到宽路径相对拆分的速度。历史单对照 4.44× 与新测三个配对分开记录。[逐点数值](../../docs/figures/manifest.json)。

## 5. Native K64 暴露了兼容性分叉

小型 64 输出分组测试使用 `[64,64,1,1]` signed INT4 权重，每个输出行有两个 K32 scale，输入为单位阵。Core ML direct signed INT4 的 native K64 资产回读数值正确，但编译计划选择 CPU；split 版本同样数值正确，也选择 CPU。更具体地，native K64 卷积的 supported 列表仅有 CPU；split 的两个卷积都列出 CPU 与 ANE 支持，却首选 `MLCPUComputeDevice`，且四个控制窗口均没有成功 ANE 请求。也就是说，“ANE supported”不能替代“ANE preferred”或“有 ANE 事件”。

Core AI native K64 LUT 路径的调用有 ANE 参与，却产生与 flattened-scale 预测一致的错误，对目标参考的 L2 约为 0.450247；K32 split 则通过精确参考并有 ANE 参与。这支持 scale-indexing 假说，但不是内部实现证明；两框架权重 IR 不同，不能归结为共同后端 bug。见[该发现](../../findings/coreai-flattened-scale/)。

历史 15360 输出、取自 Gemma 权重的 K64 片段曾触发 Core ML 诊断：ANE 只支持 per-cout/per-tensor；当前 64 输出测试是另一实验。两者不只输出通道数不同，权重和 scale 分布也不同（[历史拒绝记录](../../results/historical/coreml-native-k64-rejection.json)）。旧 K32 拆分在 Core ML 上有 ANE 参与，当前小 K32 拆分选择 CPU；这一差异可以提出新的实验问题，却不能作为仅改变 shape 的因果对照。

## 6. 三层证据与机制假说

下表汇总观察、机制假说和待检验的问题。表中的“假说”是待检验解释，不是本轮证明的事实。

| 观察 | 已证实的层次 | 合理机制假说 | 当前未知 |
|---|---|---|---|
| Core ML native K64 数值正确但 CPU-preferred | 资产和数值正确；无 ANE 准入 | 编译器对沿 K 的分组 scale 有限制 | 是量化粒度、QDQ、布局还是组合条件触发限制 |
| Core AI native K64 与 flattened-scale 预测一致 | ANE 参与与错误输出同时出现 | native LUT 的 scale 索引在 K 组边界处理不符参考 | 内部 IR 到硬件内核的具体映射 |
| Core AI 小 K32 split 数值正确且有 ANE 参与 | 拆分图的参考、控制和设备事件通过 | 每个子卷积的 per-cout scale 更容易被运行时接受 | 加法树、调度或内核选择分别贡献多少 |
| 宽路径的速度约为 split 的四倍 | 三组独立进程的同步预测计时 | 窄 K 计算、部分和归约或融合/调度成本可能增加 | 各图操作的独立成本及能否融合 |

静态解码不能证明设备执行，设备事件不能证明物理指令是 INT8，数值通过也不能证明模型质量；未通过检查的配置仍保留记录，但不纳入性能比较。

## 7. 复现与下一步

在[锁定环境](../../docs/REPRODUCING.md)中，使用尚不存在的输出目录：

```sh
.venv/bin/ane-scope run --suite split --output runs/article-split
.venv/bin/ane-scope verify runs/article-split
.venv/bin/ane-scope run --suite compatibility --output runs/article-groups
.venv/bin/ane-scope verify runs/article-groups
```

这些命令由资源保护器串行执行。数值或设备准入不通过的候选保留记录，不进入测速。只检查现有结果时，可运行 `.venv/bin/python scripts/summarize.py`，从[逐次拆分测量](../../results/fresh/split.json)重新计算表格，不需要 ANE。

后续建议在同一数据下单独改变输入 A8 QDQ、K 分区或归约方式。如果改变 scale 粒度，要先检查解码后的权重值是否保持不变；若需要重新量化，就另外报告质量变化。更广真实 Q4_0 权重、稠密激活、其他规模、芯片和版本仍需独立验证。

后续的 [G2 服务实验](04-W4A16留在ANE上值得吗.md)单独测量 W4A16 的组件速度、热响应和 GPU 共存。
