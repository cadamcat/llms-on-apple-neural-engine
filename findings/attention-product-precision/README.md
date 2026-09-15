# Small attention products lose precision on the tested ANE path

## Symptom

A weight-free attention graph with Qwen3-like proportions (32 query heads, 8 key-value heads, head size 128), 64 queries over 4,096 keys, FP16 Q, K and V from three seeds. Relative L2 is against an independent FP64 reference; the screen was 1%.

| Expression on ANE | Relative L2, three seeds |
|---|---:|
| Dense: softmax, then `P @ V` | <!-- claim:g5.random.dense-l2@g5-001 -->5.22–5.42%<!-- /claim --> |
| 1,024-key blocks, same expression | <!-- claim:g5.random.block-l2@g5-002 -->1.01–1.02%<!-- /claim --> |
| 1,024-key blocks, `exp(S − m) @ V` before normalizing | <!-- claim:g5.random.post-pv-l2@g5-003 -->0.106–0.113%<!-- /claim --> |

The same dense expression evaluated in FP16 by PyTorch on the CPU is <!-- claim:g5.cpu.dense-l2@g5-004 -->0.031%<!-- /claim --> from the reference (seed 0), so the dense error is not a necessary consequence of FP16 arithmetic.

## It is in the product of probabilities and values

A graph containing only `P @ V` and layout operations, with a uniform `P` whose nonzero entries are <!-- claim:g5.pv.probability-span@g5-005 -->0.000244–0.000248<!-- /claim -->, is **<!-- claim:g5.pv.l2@g5-006 -->3.95%<!-- /claim -->** from its reference. Its output is byte-identical to the dense graph run with zero Q and K. Scaling the operands changes the error:

| `P` fed to the same graph | Relative L2 |
|---|---:|
| `P`, as computed | <!-- claim:g5.pv.l2@g5-007 -->3.95%<!-- /claim --> |
| 16 × `P`, output divided by 16 on the CPU | <!-- claim:g5.pv.scale-16-l2@g5-008 -->0.22%<!-- /claim --> |
| 64 × `P`, output divided by 64 on the CPU | <!-- claim:g5.pv.scale-64-l2@g5-009 -->0.060%<!-- /claim --> |
| 64 × `P`, output divided by 64 inside the graph | <!-- claim:g5.pv.scale-64-l2@g5-010 -->0.060%<!-- /claim --> |

With V set to the FP16 constants 0.0005 and 0.00025, the normalized product returns **all zeros**. Multiplying unnormalized weights first and dividing afterwards gives <!-- claim:g5.constant.full-unnormalized-l2@g5-011 -->0.67%<!-- /claim --> and <!-- claim:g5.constant.half-unnormalized-l2@g5-012 -->2.37%<!-- /claim -->.

The expression that passes the random seeds also depends on magnitude. Halving that constant V puts it at <!-- claim:g5.near-zero.kv-unrolled-l2@g5-013 -->2.38%<!-- /claim --> as a single 4,096-key block and <!-- claim:g5.near-zero.post1024-l2@g5-014 -->2.36%<!-- /claim --> in 1,024-key blocks. A dense graph with `P` scaled by 512 reaches <!-- claim:g5.near-zero.dense-l2@g5-015 -->3.15%<!-- /claim -->.

Three simple FP16 models were applied to the isolated product's first query. None reproduces the device output:

| Model | From the exact result | From the ANE output |
|---|---:|---:|
| Round each product to FP16, sum wide | <!-- claim:g5.rounding.rounded-half-products-wide-sum.vs-truth@g5-016 -->0.030%<!-- /claim --> | <!-- claim:g5.rounding.rounded-half-products-wide-sum.vs-device@g5-017 -->4.04%<!-- /claim --> |
| Round each product, accumulate serially in FP16 | <!-- claim:g5.rounding.serial-half-accumulation.vs-truth@g5-018 -->0.89%<!-- /claim --> | <!-- claim:g5.rounding.serial-half-accumulation.vs-device@g5-019 -->4.19%<!-- /claim --> |
| Drop products below the smallest normal FP16 | <!-- claim:g5.rounding.products-flushed-below-half-min-normal.vs-truth@g5-020 -->17.7%<!-- /claim --> | <!-- claim:g5.rounding.products-flushed-below-half-min-normal.vs-device@g5-021 -->18.2%<!-- /claim --> |

Together with [the dot product outside its binary16 bracket](../fp16-dot-residual/), this is a second product-and-sum result that these rounding models do not explain, here at 4,096 terms and with operands in the range attention probabilities take.

## Chunking did not buy speed

Holding the expression fixed separates the effect of block size from the precision change. The same expression, multiplying unnormalized weights by V before dividing, ran as one 4,096-key block and as 1,024-key blocks inside one graph:

| Block size | Resident operation (ms) |
|---|---:|
| 4,096 keys | <!-- claim:g5.time.post4096.resident@g5-030 -->4.40<!-- /claim --> |
| 1,024 keys | <!-- claim:g5.time.post1024.resident@g5-031 -->4.35<!-- /claim --> |

The paired speed of the 1,024-key blocks is **<!-- claim:g5.speed.b1024-vs-b4096.resident@g5-032 -->1.009×<!-- /claim -->** that of the 4,096-key block for the resident operation and **<!-- claim:g5.speed.b1024-vs-b4096.with-transfer@g5-033 -->0.980×<!-- /claim -->** including layout conversion and readback. Smaller blocks gave no useful speed gain in this comparison. Resident timing includes the awaited Core AI calls and host output handling.

An earlier round compared the dense expression with the block expression that passes the random-seed precision screen. It measures the cost of that rewrite as well as blocking. Three processes, five pairs each; milliseconds per operation. The four-call path also copies K/V blocks and masks inside the resident window.

| Path | Resident operation (ms) | With layout conversion and readback (ms) |
|---|---:|---:|
| Dense (<!-- claim:g5.random.dense-l2@g5-022 -->5.22–5.42%<!-- /claim --> error) | <!-- claim:g5.time.dense.resident@g5-023 -->1.63<!-- /claim --> | <!-- claim:g5.time.dense.with-transfer@g5-024 -->9.53<!-- /claim --> |
| 1,024-key blocks, one graph | <!-- claim:g5.time.kv-unrolled.resident@g5-025 -->4.34<!-- /claim --> | <!-- claim:g5.time.kv-unrolled.with-transfer@g5-026 -->12.12<!-- /claim --> |
| 1,024-key blocks, four calls | <!-- claim:g5.time.kv-streamed.resident@g5-027 -->11.30<!-- /claim --> | <!-- claim:g5.time.kv-streamed.with-transfer@g5-028 -->13.75<!-- /claim --> |

Paired, the single-graph block path runs at **<!-- claim:g5.speed.kv-unrolled-vs-dense.resident@g5-029 -->0.376×<!-- /claim -->** the dense graph's speed for the resident operation. Host-side layout conversion and copying add <!-- claim:g5.transfer-ms-span@g5-034 -->7.6–8.1<!-- /claim --> ms to each single-graph path, more than the resident operation itself.

At 4,096 keys, blocking attention inside one graph gave no speed gain. It was not tested at the 32K graph capacity where [the complete-model path](../qwen3-4b-prefill-decode/) slows sharply. With graphs sized to each input, [most of that slowdown went away](../qwen3-4b-graph-capacity/); attention precision was not measured in that run.

## Reproduce

```sh
python scripts/verify_g5.py
```

This recomputes every timing above from <!-- claim:g5.timed-operations@g5-035 -->54,930<!-- /claim --> bundled operations and checks the paired speeds against the summaries recorded at close. The relative L2 values are imported scalars; the FP16 inputs and outputs stay with the research workspace.

## What this does not show

The inputs are synthetic: uniform and random attention, and constant values chosen to be small. Real attention is often more concentrated, and no real Q, K or V, model layer or generated text was tested. The complete-model runs in G3 were not re-evaluated against this. No per-operator placement or internal accumulation width was observed, so hardware underflow, product rounding and a compiler rewrite remain candidates. coreai-torch 0.4.2, Torch 2.9.0.

## Evidence

- [g5-attention/evidence.json](../../results/historical/g5-attention/evidence.json) — every relative L2 above; sources are listed in `provenance.json`
- [g5-attention/timings.json.gz](../../results/historical/g5-attention/) — per-operation timings
- [PROVENANCE.md](../../docs/PROVENANCE.md#g1-w-and-g5-imports) — sources and transformations
