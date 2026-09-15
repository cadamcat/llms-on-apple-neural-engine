# Related work

A summary built from source notes and URLs recorded on 2026-09-10. It is not a fresh web search or a systematic review.

## ANEMLL benchmark

[ANEMLL's recorded benchmark](https://gist.github.com/Anemll/49e219448ad350ef67ff4bfdcb9ebd8c) is an external performance reference for an M5 convolution workload. The recorded notes give roughly 17.99T Core ML FP16 and 33.86T Core ML W8A8 for it, approximately 1.88×. The INT8 experiment here uses a modified Hadamard, sign and permutation workload with repeated vectors, so it is a controlled same-shape comparison rather than a strict reproduction of the benchmark's random distribution. ANEMLL is useful context for the scale a positive result should reach; its headline is not a validation of anything in this repository.

## maderix and implementation reports

[maderix/ANE](https://github.com/maderix/ANE) reports a related M4 convolution-chain control (18.6T FP16 and 35.1T W8A8). It has code/research lineage shared with the ANEMLL example, so these are not two wholly independent replications. Its hybrid inference example uses GPU prefill and ANE decode. This is a concrete neighboring project, distinct from the CoreML-LLM conversion work below.

The recorded notes list [CoreML-LLM](https://github.com/john-rocky/CoreML-LLM) and related model conversion/runtime repositories as implementation reports. These materials discuss model packaging, block layouts, calibration, device compilation, and cases with no observed acceleration. They are practical engineering evidence with different models, APIs, versions, and hardware. They should not be collapsed into the ANEMLL benchmark or treated as a common measurement protocol.

They also list [ANEForge](https://github.com/sbryngelson/ANEForge) and its [ane-guide](https://github.com/sbryngelson/ane-guide) as a numerical and private-runtime field guide. Those sources contain boundary cases, saturation observations, cross-chip differences, and probe designs. Their private execution path and diagnostic goals differ from this project's Core ML/Core AI package checks; observations are not interchangeable.

They further list [apple/ml-ane-transformers](https://github.com/apple/ml-ane-transformers), [Anemll/Anemll](https://github.com/Anemll/Anemll), [coreai-model-zoo Gemma 12B](https://github.com/john-rocky/coreai-model-zoo/tree/main/models/gemma4-12b), and [llm-smallification](https://github.com/lube8163-lab/llm-smallification) as adjacent conversion, layout, model, and low-memory work. They establish prior art and useful comparison points.

## Interpretation boundary

ANEMLL benchmark numbers, maderix-style implementation reports, and ANEForge/ane-guide field-guide observations answer different questions. A benchmark headline is not a device trace; an implementation report is not a controlled cross-version experiment; a field guide is not a complete product compatibility matrix. This repository should retain source URLs, source snapshots, version, shape, and measurement boundary for every imported claim.

The listed sources report both positive W8A8/A8W4 controls and failures involving quantization semantics, compilation, and graph structure. The inventory was last checked on 2026-09-10 without network access, from recorded URLs. Recheck a link or a claim before relying on it.
