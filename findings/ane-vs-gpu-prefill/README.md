# Historical Python MLP: the GPU is 2.9–5.1× faster

The question most people actually have: *should I move my 4-bit model off the
GPU and onto the Neural Engine?* On this host, with this implementation, the
tested component was slower. That leaves energy, GPU coexistence and full-model
value open. The [findings](../) identify constraints, not a complete causal explanation.

## The measurement

One first-layer MLP — gate and up projections, GELU_tanh, the product, then the
down projection — built from the **same source Q4_0 weights**, with an FP16
input and output interface, run three ways. Speed, in positions processed per
second:

| Effective positions | ANE · A8W4 | ANE · W4A16 | GPU · MLX Q4 | GPU faster by |
|---:|---:|---:|---:|---:|
| 64 | 5,236/s | 5,242/s | **15,042/s** | 2.87× |
| 1024 | 5,429/s | 5,276/s | **27,622/s** | 5.09× |
| 4096 | 5,372/s | 5,304/s | **23,686/s** | 4.41× |

The same runs as latency: 12.22, 188.63 and
762.50 ms on the ANE against 4.25, 37.07 and 172.93 ms on the GPU.

Three things fall out immediately.

**This ANE implementation stays near 5,300 positions/s.** The GPU rises from
15,042 at 64 positions to 27,622 at 1024, then falls to 23,686 at 4096. The ANE
path loops fixed 64-position assets: its flat scaling does not measure the
accelerator's batch capacity.

**The gap is not inter-process overhead.** Measured inside the worker, the 4096
medians are 747.02 ms, 756.76 ms and 157.30 ms — the GPU is still 4.75× faster.
Both sides pay the same IPC, and removing it does not close the distance.

**8-bit activations showed little difference in this round.** A8W4 runs at **1.013×** W4A16 speed at 4096
positions. One process per arm does not establish equivalence or rule out an A8 benefit
in other graphs; the same-weight W4A16 path remains a useful control.

Dense-equivalent projected throughput works out at roughly 1.94 TOP/s for the
ANE path against 9.22 for the GPU. That is a component time divided into a
projection, **not** a physical operation count, and it neither confirms nor
refutes any published TOPS figure for the hardware.

## Reproduce

The recorded percentiles, ratios, footprints and completeness record are
published; recompute every ratio from them with no device:

```sh
python results/historical/tests/verify_prefill.py
```

Re-running the measurement itself needs the closed research workspace and is not
reproducible from this repository alone.

## What this is not

**It is not a statement about what the hardware can do.** These are historical
Python, 64-position-chunked measurements. The SDK limitation recorded then was
subsequently overcome. A native Swift host replaced the Python path, a persistent
pipeline followed, and a tiled-asset round measured C64/C256 with bounded memory checks.
The [follow-up record](../../results/historical/native-mlp-followup.json) keeps that progression:
C256 at 4096 positions takes **582.19 ms**, against its own round's GPU
**152.25 ms**. Both are complete external-interface timings; the rounds are not
pooled. The follow-up also has one timed process per path, not a completed
three-process benchmark. Native `run` includes runtime, scheduling and synchronization;
it is not pure device time.

**It is not an end-to-end prefill.** The 1024 and 4096 cases are formed by
looping saved real activations through one layer. There is no full-model context,
no attention, no KV cache, and no token rate here.

**It is one process per path.** Three of nine planned timed processes completed;
the run was stopped for memory safety, and the preregistered
three-process validation is incomplete. Treat the ratios as a first observation,
not a validated benchmark.

**The GPU baseline is one baseline.** MLX 0.32.2, its Q4 kernel, batched rather
than chunked — the batched form was frozen before sampling because chunking the
GPU at 64 positions cost 281 ms against 170 ms. Another backend or kernel may do
better or worse. No independent Metal kernel trace was captured.

## Why it is slow — the rest of the findings

These findings motivate hypotheses about the gap:

1. The tested Core ML direct grouped-scale representation selects CPU.
   [Evidence](../coreml-grouped-scale-cpu/).
2. A separate Core AI native LUT probe matches a flattened-scale error model.
   [Evidence](../coreai-flattened-scale/).
3. A synthetic K512 wide/split ablation incurs about four times the latency.
   [Evidence](../split-decomposition-cost/). Neither its shape nor its scale
   distribution matches this MLP, so the similar ratio is not a causal decomposition.
4. The historical Python path showed attributed memory growth; later native
   C64/C256 passed short checks. [Progress and limits](../iosurface-per-call-growth/).

A matched real-MLP ablation is still needed to connect these constraints to the
measured cost.

## Evidence

- [ane-vs-gpu-prefill.json](../../results/historical/ane-vs-gpu-prefill.json) —
  percentiles, ratios, footprints, gates and source paths
- [verify_prefill.py](../../results/historical/tests/verify_prefill.py)
- Numerical controls: all three weight tensors, 176,947,200 values, decoded on
  CPU and cross-checked against the MLX dequantization with zero numeric
  mismatches; the ANE replays match earlier runs on the same assets byte for
  byte; the GPU path's largest down-projection residual against an independent
  Torch CPU reference is 3.49e-4
- Device evidence: 72 successful ANE events per ANE path, exactly four per
  control window

## Later service experiment

[G2](../w4a16-service-tradeoffs/) adds a separate native W4A16 seven-size,
three-host comparison and finite thermal/coexistence observations. Its results
do not change this Python round's three-of-nine completion record or establish
an A8W4 improvement. The timing denominators and environments remain separate.
