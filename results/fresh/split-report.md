# Local experiment report

State: completed. Evidence verification: True.

| Process | Numeric | ANE controls | Timing | p50 ms |
|---|---|---|---|---|
| 01-control-coreai-a8w4-2 | True | True | not_requested | — |
| 02-control-coreai-a8w4-split32-2 | True | True | not_requested | — |
| 03-process1-coreai-a8w4-2 | True | True | measured | 0.431875 |
| 04-process1-coreai-a8w4-split32-2 | True | True | measured | 1.7687914999999998 |
| 05-process2-coreai-a8w4-split32-2 | True | True | measured | 1.7719369999999999 |
| 06-process2-coreai-a8w4-2 | True | True | measured | 0.4369995 |
| 07-process3-coreai-a8w4-2 | True | True | measured | 0.4455835 |
| 08-process3-coreai-a8w4-split32-2 | True | True | measured | 1.763646 |

## Independent-process summaries

| Case | Processes | p50 range ms | Source-equivalent T ops/s range |
|---|---:|---:|---:|
| coreai-a8w4-2 | 3 | 0.431875–0.445584 | 9.639–9.945 |
| coreai-a8w4-split32-2 | 3 | 1.763646–1.771937 | 2.424–2.435 |

MAC = 2 source operations. These are controlled convolution workloads, not LLM tokens/s or proof of physical INT8 instructions. Separate calls within one process are correlated. Core AI has no per-operation placement proof in this host. Energy was not measured.
