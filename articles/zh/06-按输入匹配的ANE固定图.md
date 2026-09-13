# Qwen3-4B 在 ANE 上：按输入匹配的固定图

[English](../06-qwen3-4b-matched-graphs.md) | 中文

[文章 05](05-Qwen3-4B的Prefill、Decode与能耗.md) 结尾留下一个问题。那次 ANE 路径在 256 / 2K / 32K 三个固定图中选择，一旦输入需要 32K 图，速度和每 token 能量都明显变差。这是三档图的代价，还是长输入在 ANE 上本来就如此？G4 A 保持权重、输入和 GPU 基线不变，在同一宿主实现基础上作了适配，为每档输入单独准备一个 ANE 图。

## 1. 测量的工作是什么

六档输入仍是那组 <!-- claim:g4a.contexts@g4a-001 -->500 / 1,024 / 2,048 / 4,096 / 8,192 / 16,384<!-- /claim --> token 的文档。每个 ANE 图的容量刚好容纳输入加 256 步 decode：<!-- claim:g4a.capacities@g4a-002 -->768 / 1,280 / 2,304 / 4,352 / 8,448 / 16,640<!-- /claim --> 个位置。每个图只保留自己容量的函数；每侧记录图事件的那一次请求也没有用到其他图。GPU 仍使用 <!-- claim:g4a.gpu-capacity@g4a-003 -->32,768<!-- /claim --> 位置的资产。

每侧每档输入在一个宿主会话里完成。预热请求之后静置 <!-- claim:g4a.quiet@g4a-004 -->25 秒<!-- /claim -->，运行 <!-- claim:g4a.repetitions@g4a-005 -->3<!-- /claim --> 次从 KV N 开始、各 <!-- claim:g4a.decode-steps@g4a-006 -->256<!-- /claim --> 步的 decode 请求，再静置，然后运行一组只输出 1 个 token 的 prefill 请求：六档分别为 <!-- claim:g4a.prefill-counts@g4a-007 -->130 / 90 / 55 / 34 / 15 / 7<!-- /claim --> 次，次数按此前的计时估计固定。输入从 16K 往下测到 500，两侧的先后逐档交替。

decode 使用强制续写：每次请求都接同一组 257 个 token ID，取自此前一次 GPU 运行。同一输入档位的两侧处理相同的续写序列和步数；不同档位的续写序列相同，但注意力工作量随 KV 长度增长。这里测量执行性能，不评价生成质量。预热检查首末 logits 为有限值；1K 时两侧都在记录的阈值内与独立参考一致。

能量是 CPU＋GPU＋ANE 的软件组件能量，计数器延迟假设为 <!-- claim:g4a.primary-lag@g4a-008 -->2 秒<!-- /claim -->，不扣空载。每个块单独验收：功率流完整，块结束早于最后一个功率样本至少 <!-- claim:g4a.margin@g4a-009 -->15 秒<!-- /claim -->，功率对工作有响应，并且在 0、1、2、5 秒延迟假设下时间覆盖都成立。decode 能量覆盖每次请求的首 token 到末 token，prefill 能量覆盖整块。全部 <!-- claim:g4a.blocks@g4a-010 -->24<!-- /claim --> 个块通过。运行前的短检查在每个档位都记录到宿主进程直接发出的 ANE 请求，GPU 路径上一条也没有；每个 ANE 块都有 ANE 计数器能量。这支持 ANE 确实参与执行，但不等于逐算子只在 ANE 上运行。

## 2. GPU 仍然更快

![六档输入下的 prefill 与 decode 速度，虚线为此前的三档图。](../../docs/figures/g4a-speed.svg)

两次的 decode 计量方式不同。G3 在一次请求里自由生成 1024 步；这里取它的前 256 步，并在同样的步数上积分能量，两边都覆盖 KV N 到 N + 255。G3 的 prefill 块与这里的计时方式相同。

<!-- claim:g4a.n.1024@g4a-011 -->1K<!-- /claim --> 时，ANE prefill 为 <!-- claim:g4a.prefill.1024.ane-rate@g4a-012 -->887.7 token/s<!-- /claim -->，GPU 为 <!-- claim:g4a.prefill.1024.gpu-rate@g4a-013 -->3,177.1 token/s<!-- /claim -->；从这个 KV 开始的 decode，ANE 为 <!-- claim:g4a.decode.1024.ane-rate@g4a-014 -->14.56 token/s<!-- /claim -->，GPU 为 <!-- claim:g4a.decode.1024.gpu-rate@g4a-015 -->30.09 token/s<!-- /claim -->。六档输入中，GPU 的 decode 快 <!-- claim:g4a.decode-gpu-faster@g4a-016 -->2.0–3.9×<!-- /claim -->，prefill 快 <!-- claim:g4a.prefill-gpu-faster@g4a-017 -->2.9–15.9×<!-- /claim -->。差距变大是因为 ANE 降得更多：它的 decode 从 <!-- claim:g4a.decode.500.ane-rate@g4a-018 -->15.03 token/s<!-- /claim --> 降到 <!-- claim:g4a.decode.16384.ane-rate@g4a-019 -->5.61 token/s<!-- /claim -->，GPU 从 <!-- claim:g4a.decode.500.gpu-rate@g4a-020 -->29.87 token/s<!-- /claim --> 降到 <!-- claim:g4a.decode.16384.gpu-rate@g4a-021 -->21.70 token/s<!-- /claim -->。

## 3. 三档图的代价

虚线是 G3。从 <!-- claim:g4a.n.2048@g4a-022 -->2K<!-- /claim --> 开始的 decode 越过 G3 的 2K 图，在 32K 图上为 <!-- claim:g4a.g3.decode.2048.ane-rate@g4a-023 -->3.11 token/s<!-- /claim -->；换成匹配容量的图后为 <!-- claim:g4a.decode.2048.ane-rate@g4a-024 -->13.70 token/s<!-- /claim -->，快 <!-- claim:g4a.vs-g3.decode.2048.speed@g4a-025 -->4.40×<!-- /claim -->。<!-- claim:g4a.n.4096@g4a-026 -->4K<!-- /claim --> 的 prefill 在 G3 中进入 32K 图，为 <!-- claim:g4a.g3.prefill.4096.ane-rate@g4a-027 -->84.9 token/s<!-- /claim -->；现在为 <!-- claim:g4a.prefill.4096.ane-rate@g4a-028 -->543.5 token/s<!-- /claim -->。从 <!-- claim:g4a.n.2048@g4a-029 -->2K<!-- /claim --> 起，decode 快 <!-- claim:g4a.vs-g3.long-decode-speed@g4a-030 -->1.80–4.40×<!-- /claim -->，每 token 能量为原来的 <!-- claim:g4a.vs-g3.long-decode-energy@g4a-031 -->0.42–0.68×<!-- /claim -->；从 <!-- claim:g4a.n.4096@g4a-032 -->4K<!-- /claim --> 起，prefill 快 <!-- claim:g4a.vs-g3.long-prefill-speed@g4a-033 -->2.76–6.40×<!-- /claim -->，能量为原来的 <!-- claim:g4a.vs-g3.long-prefill-energy@g4a-034 -->0.42–0.55×<!-- /claim -->。

G3 原本就能放进 2K 图的地方变化不大：<!-- claim:g4a.n.500@g4a-035 -->500<!-- /claim --> 和 <!-- claim:g4a.n.1024@g4a-036 -->1K<!-- /claim --> 的 decode 快 <!-- claim:g4a.vs-g3.short-decode-speed@g4a-037 -->1.06–1.09×<!-- /claim -->；<!-- claim:g4a.n.2048@g4a-038 -->2K<!-- /claim --> 的 prefill 速度为 G3 2K prompt 图的 <!-- claim:g4a.vs-g3.prefill.2048.speed@g4a-039 -->0.94×<!-- /claim -->。两次运行的 GPU 速度相差在 <!-- claim:g4a.vs-g3.gpu-speed@g4a-040 -->0.97–1.02×<!-- /claim --> 之内，支持图容量是主要解释。两轮没有匹配热起始，也没有隔离 ANE 路径上的其他变化。

匹配容量图消除了已观察到的长输入减速中的大部分；这组对照没有把图阶梯与两轮之间的所有其他变化逐一隔离。剩下的是随上下文变慢，匹配容量的图没有消除它，GPU 上也没有同等程度的变化。

## 4. GPU 保持工作速率，ANE 没有

![六档输入下的 prefill 推算 TFLOP/s 与 decode 推算读取 GB/s。](../../docs/figures/g4a-implied.svg)

这些速率由实测 token 速率和模型结构推算，没有测量设备计数器或内存流量，字节模型不计填充和当前 token 的写入。

prefill 每个 token 要过 <!-- claim:g4a.projection-parameters-cn@g4a-041 -->36.3 亿<!-- /claim --> 个投影权重，每个权重两个 FLOP。再加上平均 N/2 个 key 的因果注意力，GPU 在每档输入都保持 <!-- claim:g4a.gpu-total-tflops-span@g4a-042 -->20.5–27.2 TFLOP/s<!-- /claim -->：注意力变多时，每秒的投影工作大致减少同样多。ANE 则在 <!-- claim:g4a.ane-total-tflops-span@g4a-043 -->1.7–7.0 TFLOP/s<!-- /claim --> 之间下降。作为参照，同一台机器的 ANE 在一个固定形状的合成 FP16 卷积链上约为 <!-- claim:g4a.synthetic-fp16@g4a-044 -->18.8 T 源图等效 ops/s<!-- /claim -->。

decode 也是这样分开的。如果每一步读取一次 <!-- claim:g4a.weight-bytes@g4a-045 -->8.04 GB<!-- /claim --> 的 FP16 权重和已有 KV，GPU 的推算读取速率从 <!-- claim:g4a.n.500@g4a-046 -->500<!-- /claim --> 到 <!-- claim:g4a.n.16384@g4a-047 -->16K<!-- /claim --> 都在 <!-- claim:g4a.gpu-read-span@g4a-048 -->227–250 GB/s<!-- /claim --> 之间：每秒 token 变少，但每个 token 要读的 KV 变多。受带宽限制的 decode 会呈现这种形态，但这并不证明瓶颈就在带宽。ANE 的推算速率在 <!-- claim:g4a.ane-read-span@g4a-049 -->59–122 GB/s<!-- /claim --> 之间下降。

也就是说，在 ANE 上，每 token 的代价随上下文增长得比投影工作、因果注意力或 KV 读取所能解释的更快。一个候选原因在图本身：每一步 decode 计算 8 个查询位置，其中只有 1 个是真实 token，因此随 KV 增长的注意力也要为另外 7 个填充位置计算。其他候选是 ANE 上注意力每 FLOP 的代价高于投影，以及每一步的宿主工作。这些都尚未区分。在同样的图上把 decode 查询改为 1 个 token，可以去掉填充位置而不改变容量；如果那时 ANE decode 随上下文变慢的幅度不超过 GPU，主要代价就是填充位置的注意力。

## 5. 能量

![每 token 组件能量，按 CPU、GPU、ANE 计数器分段，两侧并排。](../../docs/figures/g4a-energy.svg)

柱状图为 CPU＋GPU＋ANE 的软件组件能量，不扣空载。须线描述采样时间边界，不代表传感器精度；这里没有测量整机输入电量。† 标记 CPU 功率高于相邻档的块，受影响的块见下文。

<!-- claim:g4a.short-contexts@g4a-050 -->500–2K<!-- /claim --> prefill 时，ANE 每输入 token 能量为 GPU 的 <!-- claim:g4a.short-prefill-energy-x@g4a-051 -->0.69–0.79×<!-- /claim -->。<!-- claim:g4a.long-contexts@g4a-052 -->4K–16K<!-- /claim --> 时为 <!-- claim:g4a.long-prefill-energy-x@g4a-053 -->1.05–1.24×<!-- /claim -->：匹配容量的图把长输入 prefill 从 G3 约两倍的能耗拉回到与 GPU 接近，但没有低于 GPU。decode 为 <!-- claim:g4a.decode-energy-x@g4a-054 -->0.96–1.18×<!-- /claim -->；去掉 CPU 计数器后为 <!-- claim:g4a.decode-energy-x-without-cpu@g4a-055 -->1.07–1.26×<!-- /claim -->，因为 ANE 路径的 CPU 功耗更低。

ANE 路径的 decode 窗口内，整机 GPU 计数器每 decode token 记录 <!-- claim:g4a.ane-arm-gpu-decode-energy@g4a-056 -->0.26–0.29 J/token<!-- /claim -->，占该路径 decode 能量的 <!-- claim:g4a.ane-arm-gpu-decode-share@g4a-057 -->32–58%<!-- /claim -->；ANE 计数器记录 <!-- claim:g4a.ane-arm-ane-decode-energy@g4a-058 -->0.13–0.44 J/token<!-- /claim -->。decode 变慢近三倍，这部分计数器能量按 token 归一后仍接近。但它尚未归属到具体进程或模型算子。随 token 执行的 GPU 工作是一种假设；可用等价实现替换疑似算子，检查 GPU 计数器能量是否下降，以检验这项归因。

有三个 ANE 块的 CPU 能量高于相邻档，图中以 † 标出。ANE <!-- claim:g4a.n.4096@g4a-059 -->4K<!-- /claim --> 会话期间磁盘几乎占满，并运行了排查命令：CPU 功率中位数在 decode 中为 <!-- claim:g4a.decode.4096.ane-cpu-median@g4a-060 -->1.39 W<!-- /claim -->，prefill 中为 <!-- claim:g4a.prefill.4096.ane-cpu-median@g4a-061 -->1.17 W<!-- /claim -->，而 <!-- claim:g4a.n.2048@g4a-062 -->2K<!-- /claim --> 时为 <!-- claim:g4a.decode.2048.ane-cpu-median@g4a-063 -->0.77 W<!-- /claim --> 和 <!-- claim:g4a.prefill.2048.ane-cpu-median@g4a-064 -->0.60 W<!-- /claim -->。去掉 CPU 计数器后，ANE <!-- claim:g4a.n.4096@g4a-065 -->4K<!-- /claim --> 的比值为 <!-- claim:g4a.decode.4096.energy-x-without-cpu@g4a-066 -->1.18×<!-- /claim --> 和 <!-- claim:g4a.prefill.4096.energy-x-without-cpu@g4a-067 -->0.90×<!-- /claim -->。ANE <!-- claim:g4a.n.1024@g4a-068 -->1K<!-- /claim --> decode 块的中位数为 <!-- claim:g4a.decode.1024.ane-cpu-median@g4a-069 -->1.12 W<!-- /claim -->。这些块没有重测。

## 6. 运行时的磁盘空间

这次运行差点因为与测量无关的原因停下。在这台 M5 Pro 和该 macOS 版本上，测过的 Qwen3-4B ANE 加载会留下数 GB 的编译输入文件，加载进程退出后，`ANECompilerService` 仍然开着它。每个 ANE 宿主之后，可用空间都少了 <!-- claim:g4a.disk.per-ane-host@g4a-070 -->6.8–7.0 GiB<!-- /claim --> 且不再回来，GPU <!-- claim:g4a.n.4096@g4a-071 -->4K<!-- /claim --> 在 <!-- claim:g4a.disk.gate@g4a-072 -->30 GiB<!-- /claim --> 的加载门槛前等待。结束该服务后释放了 <!-- claim:g4a.disk.released@g4a-073 -->189.1 GiB<!-- /claim -->，这些文件是两天 ANE 工作中累积下来的。等待发生在块与块之间，运行没有重启就继续了。

如果你的资产也出现同样的空间滞留，应把它计入磁盘预算，或在两次运行之间回收。[这项发现](../../findings/ane-compiler-service-disk/)给出证据，[这条 workaround](../../workarounds/#4-reclaim-disk-space-held-by-the-ane-compiler-service)提供一个在回收前检查活跃测量的定时任务。

## 7. 接下来测什么

下一轮保持这些图不变，每次只改一处：decode 查询改为 1 个 token，检验第 4 节的注意力解释；decode 改用 W4A16 权重，检验每步读取的字节变少后，ANE 是否能获得带宽模型预测的收益。把 GPU 计数器能量归属到具体进程和算子后，才能选择要测试移除的工作。在用这些曲线预测应用表现之前，还需要独立进程重复、第二颗芯片和质量评测。

[边界](../../docs/SCOPE.md#g4-a-matched-graph-observations) · [可移植证据](../../results/historical/g4a-qwen3-4b/) · [精确数值](../../docs/MEASUREMENTS.md#g4-a-qwen3-4b-fp16-with-matched-fixed-ane-graphs) · [重算方法](../../docs/REPRODUCING.md#g4-a-recomputation)
