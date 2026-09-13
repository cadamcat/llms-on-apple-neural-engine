# Is W4A16 on ANE useful? Speed, thermals and GPU coexistence

English | [中文](zh/04-W4A16留在ANE上值得吗.md)

Complete-model prefill, decode and component energy are measured separately in [article 05: Qwen3-4B FP16](05-qwen3-4b-prefill-decode-energy.md). This article covers the G2 W4A16 component run.

Article draft · 2026-09-11 · [Series index](README.md)

## Abstract

ANE can be useful even when it is slower than the GPU, if the service still meets demand while reducing sustained heat or leaving room for foreground GPU work. G2 measures a complete first-layer MLP with Q4_0 weights from the same source at seven sizes, equal arrival rates, full load and alongside foreground GPU tasks. On M5 Pro, ANE runs at about a quarter of GPU speed. At equal load both engines leave the fans at idle, while the GPU sensor reads a few degrees warmer under GPU inference; fans rise only in saturated GPU blocks. Native hosts show no per-call memory growth over finite runs, and matrix foreground work shows a smaller tail-latency penalty alongside ANE. Energy is undetermined because power capture failed integrity and clock rules. A screensaver and partially unmatched thermal starts limit attribution. The results describe component services; full-LLM benefits remain unestablished.

## 1. Speed, heat and foreground responsiveness are separate questions

The first three articles examine accelerated quantized graphs and the constraints imposed by representation, splitting and arithmetic compatibility. G2 moves to sustained service: how much work can the current W4A16 implementation handle, how does the machine heat up, and what happens to foreground GPU tasks?

Those answers need not agree. Lower instantaneous power does not guarantee lower total energy. A slower service with a modest sustained thermal load could still suit a personal computer, but lower temperatures could also reflect less completed work. The experiment therefore compares both engines running at their own capacity and both receiving the same arrival sequence and input work. The coexistence study adds a foreground-only baseline and checks whether inference requests meet their deadlines.

## 2. The workload and execution paths

The workload is Gemma 4 12B's complete first-layer MLP: gate, up, GELU_tanh/product branch and down, with H=3840 and I=15360. Both engines use Q4_0 weights from the same source and FP16 inputs and outputs. ANE uses a native Swift Core AI host; GPU uses MLX. N counts MLP positions in one request. ANE uses tile64 below N256 and tile256 otherwise, while GPU processes the whole request at once. Three saved activation inputs rotate, with some larger shapes made by repeating existing activations.

The seven sizes are 64, 128, 256, 512, 1024, 2048 and 4096 positions. Each engine uses three fresh hosts; each cell has five controls, ten warmups and thirty timed calls. Additional same-size tile diagnostics bring the total to 45 cells. Shared source weights do not require bitwise-identical backend outputs: each path passes its own frozen numerical reference before speed is compared. [Methods](../docs/METHODS.md#g2-component-service-protocol), [measurement scope](../docs/SCOPE.md#g2-service-observations) and [provenance](../docs/PROVENANCE.md#g2-native-w4a16-service).

## 3. The speed curve describes the current service implementation

![MLP throughput at seven sizes: thin lines show independent hosts and thick lines show median host rates.](../docs/figures/g2-throughput.svg)

Figure 1. The three hosts are independent repetition units; thirty calls within a host are not thirty independent experiments. One GPU round is slower at N512. The horizontal axis uses log₂ spacing and measures MLP positions throughout.

At N1024, median host rates are approximately **6,760 positions/s** for ANE and **27,300 positions/s** for GPU. ANE runs at **0.248×** the GPU speed. At N256 and above, ANE stays around 6,751–6,764 positions/s, indicating similar throughput across these request lengths under the fixed tile schedule. This does not determine whether a larger fused graph could run faster.

The N1024 tile comparison within each native host provides a narrower result: tile256 runs at **1.075–1.081×** tile64 throughput. This is a same-size diagnostic in the current run. Earlier Python and native single-process records are separate experiments, so their differences cannot be attributed entirely to a particular code change.

Saturated service uses a wider timing boundary, including validation, hashing, logging and periodic memory projections between requests. Wall-clock rates in the four blocks are **6.400 / 6.386 requests/s** for ANE, averaging 6.4, and **23.709 / 23.680 requests/s** for GPU, averaging 23.7. The ratio of the two ANE blocks' mean rate to the two GPU blocks' mean rate is **0.270×**. This exceeds the client-timed 0.248× because both engines perform the same checking and recording steps after each request, which occupy a larger share of the faster GPU service. The difference between client timing and wall-clock throughput is not direct evidence of device throttling. The original analysis separates request-time components; public recomputation uses actual completion times without subtracting estimated overhead.

## 4. Cooler at full load and cooler at equal load

![CPU and GPU temperature sensors and both fan speeds over four saturated blocks.](../docs/figures/g2-saturated-thermal.svg)

Figure 2. Each block lasts sixteen minutes, in execution order. Both pairs have matched thermal starts, and all four satisfy the recorded approximate-plateau rule over their final five minutes. GPU completes more requests, so this comparison does not establish lower energy for equal work.

In the final five minutes, median GPU-sensor readings are approximately **57.5–58.8°C** in ANE blocks and **70.0–72.0°C** in GPU blocks. Fan speeds also differ substantially. Sensor readings do not measure chassis surface temperature, and fan RPM does not measure sound pressure; neither chassis temperature nor noise was measured.

The equal-arrival-rate experiment uses the slower engine's service rate R from an independent pilot, with two paired rounds each at 0.25R, 0.5R and 0.75R. Both engines complete all **14,244** requests combined with no soft-deadline violations. All six pairs have matched thermal starts.

![Mean CPU and GPU sensor differences at equal arrival rates, subtracting ANE-inference blocks from GPU-inference blocks.](../docs/figures/g2-matched-rate-thermal.svg)

Figure 3. Over six-minute windows, the GPU-sensor mean difference is **1.33–3.29°C**, smaller than at saturation. These comparisons answer different questions, so the larger difference cannot stand in for a single cooling benefit. Most blocks do not reach an approximate plateau. Power capture exits during the first 0.75R GPU block, changing observer overhead.

The temperatures differ at equal arrival rates, but the fans do not. Across six pairs, up to **4.9 requests/s**, both engines leave mean fan speeds near the idle level at the start of each block: fan 0 is approximately **1,350 RPM**, with differences of only a few RPM. The GPU sensor reads **1.3–3.3 °C** warmer under GPU inference. Fans separate only at saturation: ANE still leaves them idle at **6.4 requests/s**, while GPU runs fan 0 at **3,100–3,500 RPM** at **23.7 requests/s**. Completion rate and execution path change together, so these blocks do not isolate either one's contribution to fan behavior at equal work. GPU loads between 4.9 and 23.7 requests/s were not measured; the load at which its fans start rising is unknown.

![Both engines leave fans at idle under equal load; only saturated GPU blocks raise fan speed.](../docs/figures/g2-load-fans.svg)

Figure 4. GPU-sensor temperature is on the left and fan 0 speed on the right. Equal-rate points use six-minute means; saturated points use medians over the final five minutes. Shading marks unmeasured GPU loads. The dashed line marks median starting fan 0 speed. Fan 1 values are given in the figure note and show the same behavior.

## 5. Coexistence needs tail measurements and a sound baseline

Foreground work consists of fixed matrix and memory-access GPU workloads. Each is calibrated alone to determine its capacity, then offered a fixed fraction of that arrival load. Every group contains foreground alone, foreground with ANE inference and foreground with GPU inference. Inference arrival rates are equal across engines, so foreground latency differences do not come from one engine receiving fewer requests.

![p95 and p99 across all eighteen foreground slots, with unmatched thermal starts marked.](../docs/figures/g2-coexistence-tails.svg)

Figure 5. Each panel has its own millisecond scale. Solid lines show p95 and dashed lines p99. Quantiles use completed work; violation rates use all arrivals. [Full counts](../docs/MEASUREMENTS.md).

Only the second 0.5F matrix group has matched thermal starts. Its foreground p95 is **30.921 ms** alone, **30.771 ms** with ANE and **34.248 ms** with GPU inference. ANE stays close to baseline while GPU inference raises the tail. Other matrix groups have the same direction, but their starting temperature and fan differences affect interpretation. Each of the twelve inference coexistence streams completes 792 requests without a soft-deadline violation.

All four matrix groups show the same pattern, under both execution orders. GPU inference raises foreground p95 by **3.3–5.6 ms** over the foreground-only baseline. With ANE, p95 is **0.144–0.151 ms** lower, an almost constant offset across groups. An additional load improving foreground speed indicates that “alone” is not a zero-interference baseline. One hypothesis is that a lightly loaded SoC remains in a lower performance state and a second workload raises it. A CPU-only busy-loop control could distinguish that effect from resource interference. The GPU penalty is more than twenty times this offset, so its direction is supported beyond the single matched group. Its magnitude may still depend on the screensaver: if animation uses GPU, it adds another competing workload whose effect could differ between inference paths.

For the memory-access foreground, deadline-violation rates in the two rounds are **13.50% / 13.37%** alone, **5.75% / 5.83%** with ANE and **12.36% / 12.34%** with GPU inference. Tails are also higher alongside GPU inference. Both foreground-only baselines need investigation; these results do not establish that ANE improves memory performance. Throughput retention near 1 at a fixed arrival rate only shows that the offered work was handled, not that all GPU capacity remains available.

## 6. Background load and screensavers on a personal computer

A resident ANE component on a personal computer shares the machine with system background work. This run measured that normal background environment.

The Ventura screensaver loaded automatically during the run, and its process appears in all 775 r4 background snapshots. Process presence does not establish rendering in every slot. It could compete directly with GPU inference or affect ANE through unified memory and thermal state, with different effects on the two paths. There was no animation-disabled control, so its overhead was neither estimated nor subtracted. A further N1024 experiment with crossed static and animated display conditions could test whether speed and foreground tails depend on that condition.

## 7. Memory behavior and remaining questions

Four fresh P0 hosts each complete 512 N4096 measurements. Each native ANE host makes **33,728** stage calls and ends with a footprint **60–62 MiB** smaller than at the start. If every gate call had retained one output buffer as in the historical Python path, gate outputs alone would have accumulated approximately **61.8 GiB**. The host, pipeline, tile size and run all differ, so this is not a binding-layer A/B test; it does show that per-call growth is not shared by every entry path.

Across 23,183 resource samples in the r4 device phase, peak component footprint is **2.928 GiB**, peak RSS across all experiment processes is **2.419 GiB**, and no swap growth is observed. The repeated RSS rises and resets during coexistence belong to the memory-access foreground, not the inference host. The component peak occurs in P2. In P0 at the same N4096, MLX hosts end at 1.55–2.32 GiB and native ANE hosts at 536 MiB. These finite runs do not establish full-model or indefinite residency, and the historical Python path remains unfixed.

Energy is undetermined. The power receiver reaches its 1 GiB capacity limit, and the complete stream also has a truncated tail and fails the original clock rules. All six equal-work energy comparisons remain undetermined. **Correction (2026-09-11):** earlier text attributed implausible system-power readings to SMC. The sampler mixes SMC and component-power estimates in this column; it cannot identify an SMC fault or substitute for the failed energy measurement. [Power measurement scope](../docs/SCOPE.md#g2-service-observations).

G2 records service throughput, thermal response and foreground interference against defined work. It supports further investigation of W4A16 services, without extending this MLP result to whole models, iPhone or other Apple chips. A8W4 remains a paused open question; this run changes neither its numerical issues nor the conditions for reopening it.

## 8. What readers can check

```sh
python scripts/verify_g2.py
python scripts/summarize.py
python scripts/render_figures.py --check
```

The [public evidence bundle](../results/historical/g2-w4a16-night/) contains the necessary request-event scalars, sensor and memory samples, natural checkpoints and source identity. Offline checks recompute the article's tables and curves. They do not repeat the original workspace's complete device and asset audits. Full original-data review and device replay still require assets and the G2 controller that are not distributed with the repository. The [reproduction guide](../docs/REPRODUCING.md#g2-recomputation-and-device-replay) distinguishes these three levels of reproduction.
