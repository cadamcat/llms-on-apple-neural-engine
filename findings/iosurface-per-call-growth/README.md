# Historical Python calls retain an output-sized IOSurface increment

The historical Python path could not safely support a long run. Two native G2 hosts
later ran 33,728 stage calls each and ended 60–62 MiB smaller; indefinite residency is
still untested.

## Symptom

Driving one gate through the Core AI Python bindings, process memory grows
monotonically with call count. Between 32 and 128 calls the IOSurface dirty size
rises by 188,743,680 bytes — **1,966,080 bytes per call**.

The gate output is 15360 channels × 64 positions ×
2 bytes = **1,966,080 bytes**. The measured increment equals one output buffer per call; it was not
released by the tested mitigations. This is attributed growth, not proof of its allocator mechanism.

Process footprints over a full measurement:

| Path | Process footprint after measurement | After 60 s idle |
|---|---:|---:|
| ANE · A8W4 | **20.21 GiB** | 19.84 GiB |
| ANE · W4A16 | **20.23 GiB** | 19.85 GiB |
| GPU · MLX | 2.24 GiB | 1.41 GiB |

The ANE paths made 161 external calls and 13,376 staged calls, loading four
assets once. The GPU path peaked at 1.33 GiB of MLX allocation and returned to
`active = 0` on release.

## Four mitigations, all ineffective

Each was a separate process with the same 128-call loop:

| Variant | Growth per call |
|---|---:|
| Ordinary NumPy read-back | 1,966,080 bytes |
| Native autorelease pool per call | 1,966,080 bytes |
| Reused input NDArray | 1,966,080 bytes |
| Direct buffer read-back, bypassing DLPack | 1,966,080 bytes |

Not reading the output into NumPy at all still grows. After closing the context
and clearing every Python reference, the buffer variant still held roughly
242 MiB of dirty IOSurface and a ~294 MiB process footprint.

## Reproduce

```sh
python results/historical/tests/verify_prefill.py
```

This recomputes the per-call growth for every diagnostic variant and asserts it
equals the gate output size. Re-running the diagnostic itself needs the closed
research workspace.

## External corroboration, and its limits

The [Pocket TTS Core AI port](https://github.com/john-rocky/coreai-model-zoo/blob/main/models/pocket-tts/README.md#the-port-in-five-core-ai-bugs)
reports the same class of problem — a per-call IOSurface leak in the Python
bindings — and works around it with call-count-limited subprocesses. That author
states a native Swift entry point does not show the same growth, and filed
FB24322437.

**That is someone else's report on someone else's model.** The same root cause
has not been established. The original SDK blocker is historical: later local
G1-HOST / PIO / TILE work introduced a persistent native Swift pipeline and
bounded its growth. [Follow-up evidence](../../results/historical/native-mlp-followup.json).

In G1-TILE, C64 and C256 passed their short memory gates. For C256, the natural
16→32 window of 4K requests grew by **16,384 bytes** in Swift and **114,688 bytes**
in the Python controller. A256 failed its separate gate. These are new observations,
not a repair of the old Python measurements and not proof of indefinite stability.

## The native path, measured

G2 later drove the same kind of component through a native Swift host. Its four P0 hosts
each served 512 measured N4096 requests, and the two ANE hosts made 33,728 stage calls each:

| P0 host | Stage calls | Footprint at start | At 512 requests |
|---|---:|---:|---:|
| ANE 1 | 33,728 | 597.8 MiB | 536.0 MiB |
| MLX 1 | — | 2,401.4 MiB | 1,591.6 MiB |
| MLX 2 | — | 2,401.2 MiB | 2,371.3 MiB |
| ANE 2 | 33,728 | 596.3 MiB | 536.2 MiB |

Both ANE hosts ended 60–62 MiB smaller than they began. Retaining one gate output per gate
call, as the Python path did, would have added about 61.8 GiB from gate calls alone. Host,
pipeline, tile size and run all differ, so this is not an A/B test of the binding; it shows
the per-call growth is not a property of every entry point.
[Service finding](../w4a16-service-tradeoffs/)

## What this does not show

The figures are process-attribution readings. They are not exclusive physical
memory, and they are not weight sizes — the true resident page attribution was
not decomposed. The system free proxy stayed at 78% or above throughout and swap
never moved beyond 2.38 MiB, so process-attributed growth and system pressure
clearly disagree, and that disagreement is unresolved.

The historical Python mechanism is not identified; the native short-window
result is a separate implementation result, not a claim of universal leak freedom. Apple's own
[autorelease pool guidance](https://developer.apple.com/library/archive/documentation/Cocoa/Conceptual/MemoryMgmt/Articles/mmAutoreleasePools.html)
supports bounding temporaries in long loops; here it did not remove the growth.

## Why it matters

A resident inference component is the ANE's plausible niche: pay the load cost
once, then answer repeatedly at low power. The observed growth blocked that historical
entry point, and that round's sustained-energy and GPU-coexistence runs were skipped.
Native G2 later ran the thermal and coexistence measurements on a host without the
growth; its energy capture failed, so energy remains undetermined.
[That separate result](../w4a16-service-tradeoffs/) does not repair the historical Python path.

## Evidence

- [ane-vs-gpu-prefill.json](../../results/historical/ane-vs-gpu-prefill.json) —
  per-variant growth, footprints, external sources and gates
- [verify_prefill.py](../../results/historical/tests/verify_prefill.py)
- [What was measured before the stop](../ane-vs-gpu-prefill/)
