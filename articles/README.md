# LLMs on Apple Neural Engine: articles

English | [中文](zh/README.md)

To decide whether to move LLM inference to ANE, start with [article 06](06-qwen3-4b-matched-graphs.md), then [article 05](05-qwen3-4b-prefill-decode-energy.md) for the earlier graph ladder. For quantization and measurement methods, start with article 01. Each article links to its Chinese version below the title.

| Article | Question |
|---|---|
| [01 · Measuring quantized ANE performance: a 35 T ops/s positive control](01-measuring-ane-performance.md)  |  What supports an acceleration measurement? |
| [02 · Group quantization on ANE: representation, compatibility and SplitConv cost](02-group-quantization-and-split.md)  |  How does preserving group scales affect execution? |
| [03 · ANE arithmetic compatibility: rounding, scale errors and model quality](03-arithmetic-compatibility.md)  |  What does a small error value establish? |
| [04 · Is W4A16 on ANE useful? Speed, thermals and GPU coexistence](04-w4a16-service-tradeoffs.md)  |  Can a slower component leave useful thermal and foreground capacity? |
| [05 · Qwen3-4B on ANE: prefill, decode and energy](05-qwen3-4b-prefill-decode-energy.md)  |  When does lower power mean lower energy for a complete LLM? |
| [06 · Qwen3-4B on ANE with a graph sized to each input](06-qwen3-4b-matched-graphs.md)  |  How much of ANE's long-input cost came from the graph ladder? |

The component findings cover [when quantization accelerates](../findings/quantized-speedup-conditions/), [QDQ multiply errors](../findings/coreai-qdq-multiply-scale/) and [attention precision and block-size comparisons](../findings/attention-product-precision/). The [compiler-service disk finding](../findings/ane-compiler-service-disk/) explains the space retained during repeated model loads on the tested machine.

For exact values and offline checks, see [measurements](../docs/MEASUREMENTS.md) and [reproduction commands](../docs/REPRODUCING.md). [Scope](../docs/SCOPE.md), [methods](../docs/METHODS.md) and [provenance](../docs/PROVENANCE.md) define what was measured and where the records came from. Figures use English labels in both versions; the [figure catalog](../docs/FIGURES.md) lists their sources. [Open research questions](../docs/RESEARCH.md).
