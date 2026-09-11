# Local experiment report

State: completed. Evidence verification: True.

| Process | Numeric | ANE controls | Timing | p50 ms |
|---|---|---|---|---|
| 01-control-coreml-fp16-2 | True | True | not_requested | — |
| 02-control-coreml-w8a8-2 | True | True | not_requested | — |
| 03-control-coreai-fp16-2 | True | True | not_requested | — |
| 04-control-coreai-w8a8-2 | True | True | not_requested | — |
| 05-control-coreai-a8w4-2 | True | True | not_requested | — |
| 06-process1-coreml-fp16-128 | True | True | measured | 15.1150205 |
| 07-process1-coreml-w8a8-128 | True | True | measured | 8.0516665 |
| 08-process1-coreai-fp16-128 | True | True | measured | 14.6620415 |
| 09-process1-coreai-w8a8-128 | True | True | measured | 7.8393335 |
| 10-process1-coreai-a8w4-128 | True | True | measured | 7.834688 |
| 11-process2-coreai-a8w4-128 | True | True | measured | 7.832791500000001 |
| 12-process2-coreai-w8a8-128 | True | True | measured | 7.8615829999999995 |
| 13-process2-coreai-fp16-128 | True | True | measured | 14.6306245 |
| 14-process2-coreml-w8a8-128 | True | True | measured | 7.9340005 |
| 15-process2-coreml-fp16-128 | True | True | measured | 14.76775 |
| 16-process3-coreml-fp16-128 | True | True | measured | 14.8732295 |
| 17-process3-coreml-w8a8-128 | True | True | measured | 8.0437295 |
| 18-process3-coreai-fp16-128 | True | True | measured | 14.641187500000001 |
| 19-process3-coreai-w8a8-128 | True | True | measured | 7.8645415 |
| 20-process3-coreai-a8w4-128 | True | True | measured | 7.837479500000001 |

## Independent-process summaries

| Case | Processes | p50 range ms | Source-equivalent T ops/s range |
|---|---:|---:|---:|
| coreml-fp16-128 | 3 | 14.767750–15.115020 | 18.186–18.613 |
| coreml-w8a8-128 | 3 | 7.934000–8.051666 | 34.139–34.646 |
| coreai-fp16-128 | 3 | 14.630624–14.662042 | 18.748–18.788 |
| coreai-w8a8-128 | 3 | 7.839334–7.864541 | 34.952–35.064 |
| coreai-a8w4-128 | 3 | 7.832792–7.837480 | 35.072–35.093 |

MAC = 2 source operations. These are controlled convolution workloads, not LLM tokens/s or proof of physical INT8 instructions. Separate calls within one process are correlated. Core AI has no per-operation placement proof in this host. Energy was not measured.
