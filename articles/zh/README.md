# LLMs on Apple Neural Engine 研究文章

[English](../README.md) | 中文

如果想判断是否把 LLM 推理挪到 ANE，先读[第 06 篇](06-按输入匹配的ANE固定图.md)，再读[第 05 篇](05-Qwen3-4B的Prefill、Decode与能耗.md)了解此前的三档图。量化与测量方法从第 01 篇读起。每篇标题下都有英文入口。

| 文章 | 主要问题 |
|---|---|
| [01 · 如何验证 ANE 的量化加速：从图表示到 35 T ops/s 正对照](01-如何验证ANE的量化加速.md)  |  什么证据足以支持一次加速测量？ |
| [02 · 分组量化如何改变 ANE 的执行图：精度、兼容性与 SplitConv 成本](02-分组量化如何改变ANE的执行图.md)  |  保留分组精度会如何影响图表示和执行成本？ |
| [03 · ANE 算术兼容性：如何区分舍入差异、scale 错算与模型质量](03-ANE算术兼容性.md)  |  一个较小的误差数字究竟说明了什么？ |
| [04 · W4A16 留在 ANE 上值得吗：速度、热行为与 GPU 共存](04-W4A16留在ANE上值得吗.md)  |  较慢的组件能否换取有用的热行为与前台空间？ |
| [05 · Qwen3-4B 在 ANE 上的 Prefill、Decode 与能耗](05-Qwen3-4B的Prefill、Decode与能耗.md)  |  完整 LLM 的低功率何时意味着低能耗？ |
| [06 · Qwen3-4B 在 ANE 上：按输入匹配的固定图](06-按输入匹配的ANE固定图.md)  |  ANE 在长输入上的代价有多少来自三档图？ |

组件实验另见[量化何时加速](../../findings/quantized-speedup-conditions/)、[QDQ 乘法错算](../../findings/coreai-qdq-multiply-scale/)与[注意力精度和分块对照](../../findings/attention-product-precision/)。[编译服务磁盘问题](../../findings/ane-compiler-service-disk/)解释了已测机器在重复加载模型时滞留的空间。

精确数值与离线检查见[测量表](../../docs/MEASUREMENTS.md)和[复算命令](../../docs/REPRODUCING.md)。[边界](../../docs/SCOPE.md)、[方法](../../docs/METHODS.md)和[来源](../../docs/PROVENANCE.md)说明测了什么、记录来自哪里。两种语言的图内均使用英文标签，来源列在[图表目录](../../docs/FIGURES.md)。[后续研究问题](../../docs/RESEARCH.md)。
