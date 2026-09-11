# LLMs on Apple Neural Engine

[English](README.md) | 中文

**4-bit LLM 值得从 GPU 挪到 Apple Neural Engine 上吗？在一台 M5 Pro 上，原生 ANE 路径跑同一个
第一层 MLP，速度约为 GPU 的四分之一。它的内存在 Python 路径会泄漏的地方保持平稳；相同负载下，
两侧的风扇都停在怠速。**

<table>
<tr>
<td><b><!-- claim:g2.ane-share@speed-card -->24.8%<!-- /claim --></b><br><sub>ANE 速度占 GPU 的比例，1024 位置，各三个宿主</sub></td>
<td><b><!-- claim:g2.native-calls@calls-card -->33,728 次调用<!-- /claim --></b><br><sub>每个原生 ANE 宿主，结束时小 <!-- claim:g2.native-shrink@memory-card -->60–62 MiB<!-- /claim --></sub></td>
<td><b><!-- claim:g2.temperature-gap@temperature-card -->1.3–3.3 °C<!-- /claim --></b><br><sub>相同负载下 GPU 推理时 GPU 传感器更高；两侧风扇都在怠速</sub></td>
<td><b><!-- claim:arithmetic.q8-summary@arithmetic-card -->13 / 7,163,904<!-- /claim --></b><br><sub>候选算术模型未匹配的最终 Q8 输出</sub></td>
</tr>
</table>

**从这里开始：** [我该用它吗？](#简短的答案) · [什么是真正能用的](workarounds/) ·
[全部发现](findings/) · [为什么慢](#为什么慢) · [复现方法](docs/REPRODUCING.md) ·
[这些数字的边界](docs/SCOPE.md) · [研究文章（English）](articles/README.md)

![G2 第一层 MLP：七档位置数下三个独立宿主的全部曲线与两侧中位数。ANE 保持在每秒约 6,760 位置，GPU 在约 18,500 到 31,600 之间。](docs/figures/g2-throughput.svg)

1024 位置时，ANE 路径约为 **<!-- claim:g2.ane-rate@ane-rate -->6,760 位置/s<!-- /claim -->**，MLX GPU 约 **<!-- claim:g2.gpu-rate@gpu-rate -->27,300 位置/s<!-- /claim -->**，均为每侧三个独立宿主的
中位数。256 位置以上 ANE 曲线持平，是因为每个请求都被切成固定的 256 位置 tile；这描述的是这条
流水线，不是硬件上限。持续服务时 ANE 为 GPU 的 **<!-- claim:g2.service-share@service-share -->27.0%<!-- /claim -->**：每个请求之后几毫秒的校验与记录两侧
相同，在更快的一侧占比更大。

![每秒完成请求数对 GPU 传感器温度与风扇 0 转速。相同负载下两侧风扇都在怠速；只有满负载 GPU 块提高了转速。](docs/figures/g2-load-fans.svg)

三档相同到达率下，最高 **<!-- claim:g2.equal-rate-max@equal-rate -->4.9 请求/s<!-- /claim -->**，两侧风扇都停在约 **<!-- claim:g2.fan-idle@fan-idle -->1,350 RPM<!-- /claim -->** 的怠速，而 GPU 推理时 GPU
传感器读数高 **<!-- claim:g2.temperature-gap@temperature-detail -->1.3–3.3 °C<!-- /claim -->**。满负载时，ANE 完成 **<!-- claim:g2.ane-saturated@ane-saturated -->6.4 请求/s<!-- /claim -->**，风扇仍在怠速；GPU 完成
**<!-- claim:g2.gpu-saturated@gpu-saturated -->23.7 请求/s<!-- /claim -->**，风扇 0 为 **<!-- claim:g2.fan-saturated@fan-saturated -->3,100–3,500 RPM<!-- /claim -->**。两者之间的 GPU 负载没有测量，GPU 风扇从哪里开始
升高仍未知。能耗未判定：功率采集没有通过完整性与时钟检查。

## 在什么机器上测的

| 机器 | 内存 | 系统 | 工具链 |
|---|---|---|---|
| **Apple M5 Pro** | 48 GiB | macOS 27.0 | G2：build 26A428 · Xcode 27.0 · MLX 0.32.2（GPU 宿主）<br>新合成套件与原生后续：build 26A428 · Xcode 27.0（Swift 6.4）；套件另有 coremltools 9.0 · coreai-torch 0.4.1 · coreai-core 1.0.0b2<br>历史 Python MLP：Xcode 26.6 · MLX 0.32.2<br>历史量化版本矩阵：build 26A5425a · coreai-torch 0.4.1 与 0.4.2 |

## 简短的答案

| 如果你的模型是…… | 在当前工具链上 |
|---|---|
| **测过的 Core ML direct signed-INT4 K64 图**，每行两个 K32 scale | 数值正确、选择 CPU。相关历史图报告 `ANE only support per-cout/per-tensor quantization`，不是所有 Q4 格式或表示的测试 |
| **K32 拆分**为按输出通道的 scale | Core AI 小型兼容探针通过；独立合成消融[慢约 4 倍](findings/split-decomposition-cost/)，不是通用代价 |
| **分组 4-bit 走 Core AI 原生 LUT 路径** | 接受、确有 ANE 活动，[并且算错](findings/coreai-flattened-scale/)——4096 个值里错 1921 个，且可预测 |
| **原生 ANE 宿主上的 W4A16 服务** | [约为 GPU 速度的四分之一](findings/w4a16-service-tradeoffs/)。相同负载下两侧风扇都在怠速；矩阵前台在 ANE 同跑时尾延迟代价更小。能耗未判定 |
| **常驻执行** | 历史 Python gate 每调用保留 1,966,080 字节；两个原生 G2 宿主各做了 33,728 次 stage 调用，结束时小 60–62 MiB；[无限期常驻未测](findings/iosurface-per-call-growth/) |
| **W4A16 对比 A8W4** | 历史 4K 配对[相差 1.3%](findings/ane-vs-gpu-prefill/)，未建立 A8 速度收益；G2 只测试 W4A16 |
| **测过的 W8A8 合成控制** | 可加速——Core AI 受控 128 层链相对自身 FP16 基线为 **1.86–1.87×** |

早期轮次作为独立记录保留。历史 Python MLP 中，64 位置时 GPU 快 **2.87×**，1024 时 **5.09×**，
4096 时 **4.41×**，每侧单进程且提前停止；之后原生 C256 在 4096 位置为 582.19 ms，同轮 GPU 为
152.25 ms，原生 `run` 时间仍包含运行时与同步。[Python 轮次](findings/ane-vs-gpu-prefill/) ·
[原生后续记录](results/historical/native-mlp-followup.json)。各数字的边界
见 [SCOPE.md](docs/SCOPE.md)。

## 仓库里有什么

四个部分，各自都能单独使用：

| 部分 | 内容 |
|---|---|
| 🔧 **三件真正能用的事** | 让分组 4-bit 能上加速器的那个改写、修好不等 scale 乘法的那个图表达、以及本来就能加速的那种量化方案——每一件都写清了代价和边界，每一件都是能跑的用例。[workarounds/](workarounds/) |
| 📊 **组件对照** | [G2](findings/w4a16-service-tradeoffs/)补入原生 W4A16 七档速度、原生宿主内存、同率与满负载温度与风扇响应、GPU 前台尾延迟。[早期 A8W4/W4A16/GPU 对照](findings/ane-vs-gpu-prefill/)保留原数值控制、按 PID 证据和停止状态；不同轮次不合并。 |
| 🐛 **可复现的缺陷** | 两个工具链失败，各有最小复现、预期错误输出和配对反向对照；一个内存泄漏，含四种无效的缓解尝试与外部佐证；一个结构性代价，三组配对进程测得。[findings/](findings/) |
| 🔬 **一个算术模型** | 候选算术模型在两个真实模型的 7,163,904 个最终 Q8 gate 输出上留下 13 处差异——以及已经定位的一个 32 项点积，不需要任何 Apple 硬件即可从公开标量验证。[模型](findings/execution-model/) · [残差](findings/fp16-dot-residual/) |

## 这个仓库适合谁

- 🟡 **想把推理从 GPU 挪走，正在判断值不值。**
  → [简短的答案](#简短的答案)，然后是[服务对照](findings/w4a16-service-tradeoffs/)及其边界。
- 🔴 **分组量化模型能编译却跑在 CPU 上，或者输出是乱的。**
  → 先看[能怎么办](workarounds/)，再看
  [Core ML 的约束](findings/coreml-grouped-scale-cpu/)与
  [Core AI 的 scale bug](findings/coreai-flattened-scale/)。
- 🟠 **长时间运行的 ANE 进程一直在涨。**
  → 历史 Python 绑定每次调用保留 [1.875 MiB](findings/iosurface-per-call-growth/)，页面里有无效的缓解尝试和外部报告；
  两个原生 Swift 宿主各做了 33,728 次 stage 调用，没有出现这种增长。
- 🔬 **在调量化数值，分不清是舍入差异还是 bug。**
  → [执行模型](findings/execution-model/)与[那个残差](findings/fp16-dot-residual/)。
- 🧪 **想复现或推翻这些结论。** → [REPRODUCING.md](docs/REPRODUCING.md)。
  已注册陈述可从随仓库记录离线核对；历史完整数组重放仍需原资产。

**状态。** 研究产物，不是受支持的产品。尚无外部复现。G2 有三个
限定条件：运行中有意外加载的动态屏保（收尾后才发现）；六个共存组中五组热起始不匹配；功率采集
失败，能耗未判定。[G2 边界](docs/SCOPE.md#g2-service-observations) · [RESEARCH.md](docs/RESEARCH.md)

## 为什么慢

以下约束已有实测；它们各自对真实 MLP 与 GPU 差距的贡献**没有隔离**：

1. **一个直接分组 scale 图选择 CPU。** Core ML 诊断针对测过的卷积表示，
   不是所有四位格式的禁令。[→](findings/coreml-grouped-scale-cpu/)
2. **Core AI 原生 LUT 探针算错。** 输出命中冻结的 flattened-scale 错误预测；
   这支持一个假说，不是内部执行追踪。[→](findings/coreai-flattened-scale/)
3. **拆分确有成本。** 独立合成 K512 消融中，十六个 K32 卷积加归约慢约四倍。
   倍率与 MLP 差距相近，不证明原因相同。[→](findings/split-decomposition-cost/)
4. **A8 尚未显示稳定速度优势。** 历史 4K 对照相差 1.3%；后续原生轮次独立保留，
   不合并估计。[→](findings/ane-vs-gpu-prefill/)
5. **候选算术模型仍有残差。** 最终 Q8 有 13 处，QDQ 前更多。一个已定位点积落在
   精确值的两个 binary16 邻居之外，仅改变最终舍入无法解释；中间乘法与累加仍未观测。
   [→](findings/fp16-dot-residual/)

## 快速开始

先选要复现的结果：下方 `ane-scope run` 命令运行合成设备套件；首页 G2 的速度、内存、温度与风扇
观察用 `python scripts/verify_g2.py` 离线重算。G2 设备重跑仍需研究工作区与模型资产。
[命令与结果对应表](docs/REPRODUCING.md#run) ·
[当前源码的设备验证状态](docs/VALIDATION.md#current-checkout-versus-the-measured-source)

```sh
uv sync --locked --extra apple
.venv/bin/ane-scope doctor
.venv/bin/ane-scope run --suite compatibility --output runs/my-groups
.venv/bin/ane-scope verify runs/my-groups
```

`compatibility` 套件复现两个工具链发现。需要 Python 3.12 与
[REPRODUCING.md](docs/REPRODUCING.md) 描述的 Apple SDK；已有缓存时给 `uv sync` 加
`--offline`。`doctor` 与 `run` 从不获取依赖、模型或数据。每个输出目录都必须是新的。

```sh
.venv/bin/ane-scope run --suite smoke      --output runs/my-smoke
.venv/bin/ane-scope run --suite throughput --output runs/my-throughput
.venv/bin/ane-scope run --suite split      --output runs/my-split
```

一个完成的 compatibility 套件里可以包含数值失败和 CPU 选中的用例。**那些就是发现本身**，不是执行错误。

## 无需设备的核对

```sh
python results/historical/tests/verify_prefill.py     # ANE/GPU 比值与泄漏
python results/historical/tests/verify_native_followup.py  # 原生后续记录
python results/historical/tests/verify_arithmetic.py  # 执行模型与那个点积
python results/historical/tests/verify_historical.py  # 导入的计时记录
python scripts/verify_g2.py                          # G2 请求、温度与资源统计
python scripts/check_source_identity.py --check       # 当前源码与运行时身份
python scripts/summarize.py                           # 已注册表格与文字数字
python scripts/render_figures.py --check              # 图、来源与生成器身份
python -m unittest discover -s tests -v               # 舍入、饱和、被篡改的证据
```

只需标准库加 NumPy，可在任意平台运行。`summarize.py` 逐处检查中英 README 首屏的具名数值引用及单位，
并检查其他已注册数字；未注册的正文仍不在检查范围内。以上命令均不运行设备。

CI 执行以上检查，还验证图的重新生成：

```sh
python scripts/render_figures.py --write
git diff --exit-code -- docs/figures
```

Git 比较要求图文件已被跟踪；尚未提交的草稿应与修改前保存的完整 SVG 文件集逐字节比较。

## 仓库结构

| 路径 | 职责 |
|---|---|
| `workarounds/` | 三件真正能用的事，含各自的代价与边界 |
| `findings/` | 每个发现一个目录：症状、复现、证据，以及哪些仍是假说 |
| `results/historical/` | 导入记录——历史对照、算术证据与独立的 G2 服务数据包 |
| `results/fresh/` | 本包三个设备套件的每一次测量调用 |
| `src/ane_scope/_coreml.py`、`_coreai.py` | 导出适配器与持久化图/权重审计 |
| `src/ane_scope/references/` | 确定性 fixture 与显式算术参考 |
| `src/ane_scope/native/` | Swift 预测宿主与可观测的运行时元数据 |
| `src/ane_scope/controller.py`、`evidence.py`、`guard.py` | 准入、输出/放置检查与资源限制 |
| `scripts/summarize.py`、`render_figures.py` | 重算已发布数字；重新生成图 |
| `articles/` | 四篇研究文章，英文与中文互相链接 |
| `docs/` | 边界、方法、来源、复现、相关工作与开放问题 |
| `runs/` | 已忽略：本地 fixture、模型、日志、编译产物与原始输出 |

## 更正记录

- **（2026-09-11）G2 的第一版总结写过 ANE 运行时风扇转速较低。** 这只在满负载时成立，而那时 GPU
  完成的工作量约为 ANE 的 3.7 倍。相同负载下，两侧风扇都停在怠速。
- **G2 的功率采集失败。** 接收器触及 1 GiB 上限，数据流也没有通过时钟对齐规则，六组同工作量
  能耗比较全部未判定。
- **G2 运行中有动态屏保。** Ventura 动态屏保在运行中自动加载，收尾后才发现，775 个背景快照都有
  它的进程。没有关闭动画的对照，它对两侧的影响未知。
- **prefill 对照并不完整。** 九个计划中的计时进程只完成了三个，
  运行因内存保护被停止。这些比值是首批观测，不是通过验收的 benchmark。
- **一个 128 层 W8A8 控制最初以相对 L2 0.233 未通过参考。** 冻结参考用了"中点取偶"，
  而设备按"中点远离零"舍入。之后先冻结新参考、新的留出输入和原有阈值，
  再重跑实验。这次更正后来就成了[执行模型](findings/execution-model/)。
- **严格的跨模型逐字节门未通过。** E2B 完全一致，Qwen8B 余 13 处残差。
- **历史拆分比值是 4.44×，新测是 3.96–4.10×。** 不同的运行，一个进程对三个进程。
  两者不合并，差异也不归因于任何软件改动。
- **第一次 smoke 只完成了 14 个用例中的 12 个。** 两个 Core AI 分组导出被审计解析器拒绝——
  它当时还不能处理一个十六进制常量和一个开放式 slice。解析器修好后重跑。
- **Core AI 的 requested compute unit 从来不是逐算子证明。** 一次外部代码评审指出了这一点，
  现在字段显式记录这条限制。
- **早期的图声明了大多数读者没有的字体**，除了生成它们的那台机器之外到处退化成衬线体。
  现在的图以纯 SVG 文本加系统字体栈生成，且重新生成是逐字节可复现的。

## 什么会改变这个答案

下一步优先测完整模型的 prefill：相同模型、精度和 token 输入，包含 attention 与 KV cache，先通过数值检查，再比较完成相同工作量的速度、平均功率与焦耳数。两侧各自按实际速度运行。能耗比较之前，先用短先导测试验收功率采集。在第二颗 Apple 芯片上复现，以及在同一宿主中常驻数小时，可以检验结果能否迁移和持续。

后续服务实验再测 GPU 风扇在 4.9 到 23.7 请求/s 之间从哪里开始升高，先覆盖 4.9 到约 6.4 请求/s；以及矩阵前台的尾延迟优势在关闭动画、匹配热起始并加入只占 CPU 的忙等基线后是否仍然存在。[RESEARCH.md](docs/RESEARCH.md) · [RELATED_WORK.md](docs/RELATED_WORK.md)

## 贡献与许可

[CONTRIBUTING.md](CONTRIBUTING.md) · [LICENSE](LICENSE) · [NOTICE](NOTICE) ·
[代码来源](docs/code-origins.json)。项目代码与生成的实验记录使用 MIT；依赖保留各自的许可，
不再分发任何模型权重、Apple 框架二进制或编译后的模型。这是一个独立项目，与 Apple 无关。
