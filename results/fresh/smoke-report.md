# Local experiment report

State: completed. Evidence verification: True.

| Process | Numeric | ANE controls | Timing | p50 ms |
|---|---|---|---|---|
| 01-control-coreml-fp16-2 | True | True | not_requested | — |
| 02-control-coreml-w8a8-2 | True | True | not_requested | — |
| 03-control-coreai-fp16-2 | True | True | not_requested | — |
| 04-control-coreai-w8a8-2 | True | True | not_requested | — |
| 05-control-coreai-a8w4-2 | True | True | not_requested | — |
| 06-control-coreai-a8w4-split32-2 | True | True | not_requested | — |
| 07-control-coreml-group-native64 | True | False | not_requested | — |
| 08-control-coreml-group-split32 | True | False | not_requested | — |
| 09-control-coreai-group-native64 | False | True | not_requested | — |
| 10-control-coreai-group-split32 | True | True | not_requested | — |
| 11-control-coreai-qdq-equal | True | True | not_requested | — |
| 12-control-coreai-qdq-unequal | False | True | not_requested | — |
| 13-control-coreai-qdq-reverse_order | False | True | not_requested | — |
| 14-control-coreai-qdq-explicit_q | True | True | not_requested | — |

## Independent-process summaries

| Case | Processes | p50 range ms | Source-equivalent T ops/s range |
|---|---:|---:|---:|

MAC = 2 source operations. These are controlled convolution workloads, not LLM tokens/s or proof of physical INT8 instructions. Separate calls within one process are correlated. Core AI has no per-operation placement proof in this host. Energy was not measured.
