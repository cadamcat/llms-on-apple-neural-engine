# Small attention products lose precision on the tested ANE path

## Symptom

A weight-free attention graph with Qwen3-like proportions (32 query heads, 8 key-value heads, head size 128), 64 queries over 4,096 keys, FP16 Q, K and V from three seeds. Relative L2 is against an independent FP64 reference; the screen was 1%.

| Expression on ANE | Relative L2, three seeds |
|---|---:|
| Dense: softmax, then `P @ V` | 5.22–5.42% |
| 1,024-key blocks, same expression | 1.01–1.02% |
| 1,024-key blocks, `exp(S − m) @ V` before normalizing | 0.106–0.113% |

The same dense expression evaluated in FP16 by PyTorch on the CPU is 0.031% from the reference (seed 0), so the dense error is not a necessary consequence of FP16 arithmetic.

## It is in the product of probabilities and values

A graph containing only `P @ V` and layout operations, with a uniform `P` whose nonzero entries are 0.000244–0.000248, is **3.95%** from its reference. Its output is byte-identical to the dense graph run with zero Q and K. Scaling the operands changes the error:

| `P` fed to the same graph | Relative L2 |
|---|---:|
| `P`, as computed | 3.95% |
| 16 × `P`, output divided by 16 on the CPU | 0.22% |
| 64 × `P`, output divided by 64 on the CPU | 0.060% |
| 64 × `P`, output divided by 64 inside the graph | 0.060% |

With V set to the FP16 constants 0.0005 and 0.00025, the normalized product returns **all zeros**. Multiplying unnormalized weights first and dividing afterwards gives 0.67% and 2.37%.

The expression that passes the random seeds also depends on magnitude. Halving that constant V puts it at 2.38% as a single 4,096-key block and 2.36% in 1,024-key blocks. A dense graph with `P` scaled by 512 reaches 3.15%.

Three simple FP16 models were applied to the isolated product's first query. None reproduces the device output:

| Model | From the exact result | From the ANE output |
|---|---:|---:|
| Round each product to FP16, sum wide | 0.030% | 4.04% |
| Round each product, accumulate serially in FP16 | 0.89% | 4.19% |
| Drop products below the smallest normal FP16 | 17.7% | 18.2% |

Together with [the dot product outside its binary16 bracket](../fp16-dot-residual/), this is a second product-and-sum result that these rounding models do not explain, here at 4,096 terms and with operands in the range attention probabilities take.

## Chunking did not buy speed

The accurate block expression costs more time than the inaccurate dense graph. Three processes, five pairs each; milliseconds per operation.

| Path | Kernel call only (ms) | With layout conversion and readback (ms) |
|---|---:|---:|
| Dense (5.22–5.42% error) | 1.63 | 9.53 |
| 1,024-key blocks, one graph | 4.34 | 12.12 |
| 1,024-key blocks, four calls | 11.30 | 13.75 |

Paired, the single-graph block path runs at **0.376×** the dense graph's speed for the kernel call. In a later round the same accurate expression as one 4,096-key block and as 1,024-key blocks ran at 4.40 and 4.35 ms: blocks gain **1.009×** on the kernel call and **0.980×** including transfer. Host-side layout conversion and copying add 7.6–8.1 ms to each single-graph path, more than the kernel call itself.

At 4,096 keys, blocking attention inside one graph gave no speed gain. It was not tested at the 32K graph capacity where [the complete-model path](../qwen3-4b-prefill-decode/) slows sharply.

## Reproduce

```sh
python scripts/verify_g5.py
```

This recomputes every timing above from 54,930 bundled operations and checks the paired speeds against the summaries recorded at close. The relative L2 values are imported scalars; the FP16 inputs and outputs stay with the research workspace.

## What this does not show

The inputs are synthetic: uniform and random attention, and constant values chosen to be small. Real attention is often more concentrated, and no real Q, K or V, model layer or generated text was tested. The complete-model runs in G3 were not re-evaluated against this. No per-operator placement or internal accumulation width was observed, so hardware underflow, product rounding and a compiler rewrite remain candidates. coreai-torch 0.4.2, Torch 2.9.0.

## Evidence

- [g5-attention/evidence.json](../../results/historical/g5-attention/evidence.json) — every relative L2 above; source hashes are in `provenance.json`
- [g5-attention/timings.json.gz](../../results/historical/g5-attention/) — per-operation timings
- [PROVENANCE.md](../../docs/PROVENANCE.md#g1-w-and-g5-imports) — sources and transformations
