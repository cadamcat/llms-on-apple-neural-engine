# LLMs on Apple Neural Engine: articles

English | [中文](zh/README.md)

The articles connect complete-model speed and energy with the component experiments: when ANE accelerates quantized graphs, how representation and arithmetic constrain that acceleration, and whether a slower ANE service can leave useful room for foreground GPU work. Each article has a complete Chinese version linked below its title.

If you are deciding whether to move inference to ANE, start with [article 05](05-qwen3-4b-prefill-decode-energy.md). For the measurement method and quantization results, read the series in order. These are research drafts based on the bundled experiments: preparing articles 01–03 added no device runs, and article 04 uses G2, and article 05 uses the complete-model G3 measurements.

## Reading order

| Article | Question | Main evidence |
|---|---|---|
| [01 · Measuring quantized ANE performance: a 35 T ops/s positive control](01-measuring-ane-performance.md) | What supports an acceleration measurement? | A full 128-layer chain, five configurations and three independent processes each; assets, numerical controls, ANE participation and timing |
| [02 · Group quantization on ANE: representation, compatibility and SplitConv cost](02-group-quantization-and-split.md) | How does preserving group scales affect execution? | Paired wide-K and K32 split measurements; correctness and device selection for native and split grouped graphs |
| [03 · ANE arithmetic compatibility: rounding, scale errors and model quality](03-arithmetic-compatibility.md) | What does a small error value establish? | Frozen RNE/RZA references, exact-grid QDQ and a grouped-scale error prediction |
| [04 · Is W4A16 on ANE useful? Speed, thermals and GPU coexistence](04-w4a16-service-tradeoffs.md) | Can a slower component leave useful thermal and foreground capacity? | G2 at seven sizes in three hosts per engine, equal-rate and saturated service, and all eighteen coexistence slots |
| [05 · Qwen3-4B on ANE: prefill, decode and energy](05-qwen3-4b-prefill-decode-energy.md) | When does lower power mean lower energy for a complete LLM? | Same-model FP16 prefill and decode, warmed stage blocks and software component energy |

Article 01 defines the measurement. Article 02 connects graph representation with performance, and article 03 separates the different questions called “accuracy.” Article 04 moves native W4A16 into component service, thermal response and GPU coexistence.

Figures use English labels in both versions, with captions and alternative text in the article's language. The [figure catalog](../docs/FIGURES.md) records source data, point values, scope and generation instructions.

## Conclusions and their evidence

| Conclusion | Evidence | Scope |
|---|---|---|
| A synthetic wide-K quantized chain accelerates reproducibly on the tested path | [Fresh throughput](../results/fresh/throughput.json): `runs[].measured_records`, `admission`, `numerical` | T ops/s uses source-graph work; it does not establish physical INT8, quality equivalence or full-LLM benefits |
| The two-layer wide-K path runs at about four times K32 split speed with the same weights | [Fresh split records](../results/fresh/split.json): paired p50 from three independent-process rounds | Decomposition, reduction and rounding boundaries change together; their individual costs are not isolated |
| Correct persisted weights, correct output and ANE execution need separate checks | [Fresh smoke](../results/fresh/smoke.json): `asset_audit`, `numerical`, `placement` | Core AI lacks per-operation placement evidence; framework representations differ |
| An explicit rounding reference explains some long-chain discrepancies | [Historical arithmetic excerpt](../results/historical/arithmetic-reference-evidence.json) and fresh throughput `RNE-not-gate` | The device output is unchanged; a newly selected reference needs frozen held-out validation |
| Exact-grid inputs still expose scale- and expression-dependent errors | Fresh smoke `coreai-qdq-*` and `coreai-group-*` | Predicting the error does not prove the compiler implementation; fresh tests cover only 0.4.1 |
| A candidate arithmetic model leaves few residuals in final Q8 outputs from two real models | [Cross-model records](../results/historical/cross-model-validation.json) and [dot localization](../results/historical/dot-localization.json) | Final Q8 has 13 residuals, with more before QDQ; the mechanism remains a hypothesis |
| The native W4A16 component runs at about a quarter of GPU speed | [G2 bundle](../results/historical/g2-w4a16-night/): seven sizes, three hosts per engine | One machine and its first-layer MLP; a fixed tile pipeline, not a hardware ceiling |
| Native hosts avoid the Python path's observed per-call memory growth | G2 `p0.json` natural checkpoints | Finite runs; indefinite residency is untested |
| At equal load both engines leave fans idle, with sensor readings a few degrees apart | G2 six paired equal-arrival-rate groups | Six-minute transient windows; GPU loads between 4.9 and 23.7 requests/s are unmeasured |
| Matrix foreground tails have a smaller penalty alongside ANE | G2 eighteen coexistence slots | Only one group has matched thermal starts; the screensaver is uncontrolled and the foreground-only baseline has an offset |

[MEASUREMENTS.md](../docs/MEASUREMENTS.md) is generated from bundled per-call records by [scripts/summarize.py](../scripts/summarize.py). Historical records are labeled `historical_import` and suite records `fresh_run`. Articles use rounded values; the linked records contain the precise fields.

## Maintaining the series

`articles/` explains experimental motivation, design, results and inference. [SCOPE.md](../docs/SCOPE.md) defines evidence boundaries, [METHODS.md](../docs/METHODS.md) specifies measurement, [REPRODUCING.md](../docs/REPRODUCING.md) gives environments and commands, [PROVENANCE.md](../docs/PROVENANCE.md) records sources, and [RESEARCH.md](../docs/RESEARCH.md) lists open questions.

Keep English and Chinese articles aligned when a claim changes, including its evidence, qualification and numeric registration. Check the relevant records and code first. Toolchain fixes belong in new observations. Changes to inputs, references, thresholds or graph structure define a new experiment and need a new result directory; historical records stay unchanged.

Further development into a paper needs independent device replication, stronger experiments that change one factor at a time and broader input distributions. G3 adds complete-model performance and software component energy. Broader model-quality evaluation, calibrated energy and transfer across devices remain open. [Further research](../docs/RESEARCH.md).
