# One- and two-position decode queries fail on ANE

## Symptom

A static-shape Qwen3-4B export for ANE through coreai-models specializes its transformer graph for fixed query widths. With functions for 1, 8 and 64 query positions in one bundle, prompt and eight-position requests run on ANE, but the first request to the one-position decode function fails:

```text
ANE_ProgramSendRequest:1029 status=e00002c2
ANEProgramProcessRequestDirect() Failed with status=0x2 : statusType=0x9
ANERegion.mm:414: failed assertion `ANE inference operation failed. Error Domain=com.apple.metalperformanceshadersgraph Code=-19
```

The host process aborts. `0xe00002c2` is the IOKit invalid-argument code. The MPSGraph message reports the ANE model's earlier successful requests and no earlier failures: 19 in the 36-layer model after a 1,024-token prompt.

## Isolation

Each row is a new host process on one M5 Pro. The prompt runs through 64-position functions; decode then switches to the listed width.

| Bundle | Decode functions in the bundle | Width run | Outcome |
|---|---|---:|---|
| 36 layers, capacity 1,280 (the G4 A tier) | 1, 8, 64 | 8 | runs |
| same | same | 1 | fails after 19 successful requests |
| 1 layer, capacity 768 | 1, 8, 64 | 8 | runs |
| same | same | 1 | fails after 2 |
| 1 layer, capacity 768 | 1, 64 | 1 | fails after 2 |
| 1 layer, capacity 768 | 1, 2, 4, 8, 64 | 4 | runs; same greedy tokens and logits as width 8 |
| same | same | 2 | fails after 2 |

So the failure does not need the full model, a long prompt, a query-8 function in the same bundle or a decode position aligned to 8. Width 1 failed in the {1, 8, 64} and {1, 64} bundles. In the {1, 2, 4, 8, 64} bundle, width 2 failed while widths 4 and 8 ran; width 1 was not run in that bundle. The input descriptors of the failing functions scale with the width like those of the working ones: `transformer_input` 1 × q × 1 × 2560, `position_ids` 1 × q and `causal_mask` 1 × C × 1 × q.

Tracing the graph at width 1 instead of the exporter's default 8 does not produce a bundle: `torch.export` reports that `seq_len` was marked dynamic but specialized to the constant 1.

## Boundary

The coreai-models exporter specializes query widths 8, 16 and 64 by default; widths 1, 2 and 4 were added here through its class attribute. The failure is therefore on shapes outside the exporter's default set, reached with a two-line change to the host's allowed widths. Which operator or input the ANE request rejects, and whether width 3 runs, were not tested. G6 used width 4 to measure the effect of padded query positions on complete-model decode ([result](../qwen3-4b-graph-capacity/#repeat-and-decode-query-width)).

Environment: Apple M5 Pro, macOS 27.0 (26A428), coreai-models 7304c47, coreai-torch 0.4.2, Torch 2.9.0.

## Reproduce

The [recorded outcomes](repro/recorded/) are checked offline:

```sh
python findings/ane-short-decode-query/repro/verify.py
```

The verifier checks every case's function inventory, host return code, ANE request and failure log counts, MPSGraph assertion and same-logits comparison. The [repro README](repro/README.md) gives the export and device commands; they need the pinned Qwen3-4B source, the coreai-models checkout and its Swift host.
