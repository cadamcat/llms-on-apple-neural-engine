# Research direction and open questions

The [findings](../findings/) measure a component gap and identify constraints;
they do not yet connect those constraints into a causal explanation. In the historical
Python MLP comparison, the GPU was 2.9–5.1× faster. Later native C256 measured 582.19 ms
against its same-round GPU 152.25 ms at 4K. Those earlier rounds each have one timed process per
path; they are not pooled. G2 now adds a separate three-host, seven-size
comparison and finite thermal/coexistence observations. [Native follow-up](../results/historical/native-mlp-followup.json).

The tested implementation remains slower. What stays open is **whether it earns
its place through lower sustained power or better GPU coexistence**, and whether
those benefits survive a complete model and a quality-matched comparison.

![Runtime compatibility, deviation from FP16 and model quality need three separate comparisons.](figures/arithmetic-levels.svg)

## The engineering position

A resident W4A16 component: the model stays loaded, the load cost is paid once,
and what matters is steady state. That fits what the accelerator is plausibly
good at — sustained low-power execution of a fixed graph — rather than peak
throughput.

The historical Python entry point showed output-sized IOSurface growth. Native
G1-HOST / PIO / TILE work subsequently changed the host and persistent pipeline;
C64/C256 passed bounded memory checks. For C256, the natural 16→32 window of 4K
requests grew by 16,384 Swift bytes and 114,688 Python-controller bytes. That
advances the entry point, not the long-term claim. The later
[G2 run](../findings/w4a16-service-tradeoffs/) completed finite memory and service
blocks and measured thermal response and coexistence; its two native ANE hosts ran
33,728 stage calls each without the Python per-call growth. Energy remains
undetermined, and full-model residency still needs measurement. A256 failed its own
short memory gate and must remain a separate result.

## The three questions that decide it

| Question | The comparison | What is measured |
|---|---|---|
| **Does it meet the need at all?** | Same model, same inputs, quality checked first, ANE against GPU | Time to first token, sustained generation rate, tail latency, memory stability — [G2 component observations exist](../findings/w4a16-service-tradeoffs/); no full-model comparison is established |
| **Does it lower energy and thermal pressure?** | Equal completed work first; then equal service rate | J/request, J/token, mean power, sustained temperature and fan state — G2: at equal load fans stayed at idle on both engines; energy undetermined |
| **Does it actually free the GPU?** | Foreground task alone, then with GPU inference, then with ANE inference | Foreground throughput or frame time — read the **p99, not the mean** — with the LLM's own response recorded alongside |

The next experiment should address speed and energy together on a complete model.
Use the same weights, precision, token input and cache policy, check outputs first,
then let each engine complete the same work at its own speed. Report prefill time,
mean power and joules per request together: lower power does not imply lower energy
when execution takes longer. Equal-rate service and foreground coexistence remain
separate questions about background use.

The objectives interact but remain distinct. Lower sustained power can still
matter without a coexistence win, even when longer execution costs more joules.
Report service rate, energy, power and thermal behaviour together.

## Method traps already identified

- **Energy needs a declared work and quality match.** Freeze actual input/output
  token budgets and cache policy; record all completed, dropped and unfinished
  requests. Different generated text is not automatically unequal work, but
  different lengths or task quality need explicit treatment. Greedy decoding and
  a seed alone do not guarantee identical output.
- **Unified memory resists attribution.** "ANE energy" may not be separable from
  whole-package power. Verify what `powermetrics` actually reports on the host
  before designing around it, and report a whole-machine delta if that is all
  there is.
- **Mean frame time hides the effect.** Stutter is what offloading is supposed to
  prevent, so the metric has to be a tail.

## What a publishable version still needs

The [findings](../findings/) include measured behaviour and mechanism hypotheses. Four gaps stand
between them and a systems paper, in dependency order:

0. **Validate power capture and move beyond the component.** Start with a short
   complete-model prefill pilot including attention, KV cache and inter-layer
   transfer. Check outputs and device placement, sampling coverage and clocks,
   counter response and observer overhead before a longer energy run. Keep the
   screensaver off for this comparison; a difference from G2 alone cannot isolate
   its effect. Crossed display conditions, matched matrix-tail starts and the
   anomalous memory-access baseline need their own controls. Native timing alone
   does not isolate hardware cost.
1. **Cross-chip replication.** One chip, one OS, one toolchain version is the
   first thing a reviewer will object to. The arithmetic findings are about
   *semantics*, which is the part most likely to transfer, and testing them
   elsewhere needs a new controlled replay of the frozen assets. This is
   the cheapest high-value work available.
2. **A comparison baseline.** "Inefficient" needs a denominator. Same model on
   GPU (MLX or Metal) and CPU, same tokens, same quality gate.
3. **Single-variable isolation of the split cost.** A preregistered ablation over
   K, reduction depth, output-channel count and representation, each varied
   alone, with effects that survive order reversal, and plan selection
   distinguished from arithmetic change.

Two smaller ones: the private Core AI interfaces the attribution depends on are
version-pinned and will age, so external symptoms and internal attribution must
stay separable in any write-up; and the residual mechanism is still a
[stated hypothesis](../findings/fp16-dot-residual/), which needs a rule fixed in
advance that explains every deviation in the block and introduces none.

## Where the returns have run out

The localization already reached a single 32-term dot product. Going deeper into
that one residual buys very little now; **breadth is what is missing** — another
chip and a complete-model baseline. G2 supplies a completed component run,
with thermal/coexistence observations whose display sensitivity remains to be tested. Any proposal for a new experiment is worth
judging on one question: which uncertain mechanism does it distinguish?

See [RELATED_WORK.md](RELATED_WORK.md) for the neighbouring projects this should
be compared against, and [HISTORICAL.md](HISTORICAL.md) for the imported records
that preceded this package.

## What G2 changes

The service question now has data: ANE ran at about a quarter of the GPU's
speed; at equal load both engines kept the fans at idle while the GPU sensor read
a few degrees warmer; and a matrix foreground showed a smaller tail penalty beside
ANE inference. The missing energy measurement and the component-only workload now
limit the decision most. A complete-model prefill comparison can show whether the
slower path uses less energy for the same work. The GPU fan threshold remains open,
as does whether the tail signal survives the display control, matched thermal
starts and a CPU-only busy baseline. Normal PC background remains part of the
target environment. The [fourth article](../articles/04-w4a16-service-tradeoffs.md)
separates measured responses from remaining hypotheses.

A8W4 remains a paused Open Question. Reopen it for a relevant upstream change,
a new representation that passes existing controls, or a test that distinguishes
arithmetic mechanisms. G2 provides no new A8 repair evidence.
