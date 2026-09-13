# Research direction and open questions

The [findings](../findings/) measure a component gap and identify constraints;
they do not yet connect those constraints into a causal explanation. In the historical
Python MLP comparison, the GPU was 2.9–5.1× faster. Later native C256 measured 582.19 ms
against its same-round GPU 152.25 ms at 4K. Those earlier rounds each have one timed process per
path; they are not pooled. G2 now adds a separate three-host, seven-size
comparison and finite thermal/coexistence observations. [Native follow-up](../results/historical/native-mlp-followup.json).

G3 now adds complete Qwen3-4B FP16 speed and software component energy. ANE saves component energy on short prefill, while the current larger-context path is slower and uses more energy per token. The next questions concern graph shape, broader quality and application behaviour. [G3 results](../findings/qwen3-4b-prefill-decode/).

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
| **Does it meet the need at all?** | Same model, same inputs, quality checked first, ANE against GPU | Time to first token, sustained generation rate, tail latency, memory stability — [G2 component observations exist](../findings/w4a16-service-tradeoffs/); G3 and G4 A provide complete-model speed over their input range; wider input and quality coverage remain open |
| **Does it lower energy and thermal pressure?** | Equal completed work first; then equal service rate | J/request, J/token, mean power, sustained temperature and fan state — G2: at equal load fans stayed at idle on both engines; energy undetermined |
| **Does it actually free the GPU?** | Foreground task alone, then with GPU inference, then with ANE inference | Foreground throughput or frame time — read the **p99, not the mean** — with the LLM's own response recorded alongside |

G4 A held model, input, precision and GPU baseline fixed and gave each input a matched ANE graph: most of G3's long-input penalty went away and the GPU stayed faster ([finding](../findings/qwen3-4b-graph-capacity/)). Two questions follow from it. ANE still slows with context more than its modelled work explains; a one-token decode query on the same graphs would test whether the seven padded query positions per step are the cause. The system GPU counter records a large share of the component energy during ANE-path decode; its process and operation sources are unassigned. Equal-rate service and foreground coexistence remain
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

0. **Extend the complete-model evidence.** G3 and G4 A have intact power capture and complete FP16 model stages, the latter with matched ANE graphs. Test the decode query width and quantized decode weights on those graphs, additional independent hosts and inputs, broader output quality, estimator latency and observer overhead. A difference from G2 alone cannot isolate the screensaver effect. Crossed display conditions, matched matrix-tail starts and the
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
chip and a controlled expansion of the complete-model baseline. G2 supplies a completed component run,
with thermal/coexistence observations whose display sensitivity remains to be tested. Any proposal for a new experiment is worth
judging on one question: which uncertain mechanism does it distinguish?

See [RELATED_WORK.md](RELATED_WORK.md) for the neighbouring projects this should
be compared against, and [HISTORICAL.md](HISTORICAL.md) for the imported records
that preceded this package.

## What G2 changes

The service question now has data: ANE ran at about a quarter of the GPU's
speed; at equal load both engines kept the fans at idle while the GPU sensor read
a few degrees warmer; and a matrix foreground showed a smaller tail penalty beside
ANE inference. G3 subsequently adds complete-model prefill/decode speed and software energy; its context-dependent results do not turn this component experiment into a full-model coexistence test. The GPU fan threshold remains open,
as does whether the tail signal survives the display control, matched thermal
starts and a CPU-only busy baseline. Normal PC background remains part of the
target environment. The [fourth article](../articles/04-w4a16-service-tradeoffs.md)
separates measured responses from remaining hypotheses.

A8W4 remains a paused Open Question. Reopen it for a relevant upstream change,
a new representation that passes existing controls, or a test that distinguishes
arithmetic mechanisms. G2 provides no new A8 repair evidence.

## What G1-W and G5 change

G1-W tested a released per-channel A8W4 checkpoint, Gemma 4 E4B mobile QAT. Its
first MLP is numerically wrong through [a QDQ multiply that uses another QDQ's
scale](../findings/coreai-qdq-multiply-scale/), and eight repeated copies gain no
speed with or without the clamp that removes the gross error. Precision fixes that
split the graph avoid a compile failure but run several times slower. On synthetic
chains the [A8 gain needs depth and depends on exact-zero weights](../findings/quantized-speedup-conditions/).
Open: why a call of E4B size gains nothing when a synthetic chain of equal work
does. Weight count per layer, the gated multiply and the shape are not yet separated.

G5 found that [small attention products lose precision](../findings/attention-product-precision/)
on the tested path, and that blocking attention inside one 4,096-key graph gives no
speed gain. Whether real attention distributions reach the failing range, and how
graph capacity rather than attention arithmetic drives the complete-model slowdown,
are open. G4 A measured the complete model at matched graph capacities: most of the observed slowdown went away, though the comparison did not isolate every change between runs. The remaining slowdown has not been attributed to attention arithmetic.

Running ANE experiments exposed an operational question: [ANECompilerService retained deleted compile inputs](../findings/ane-compiler-service-disk/) until it exited on the tested machine and assets. A minimal reproduction with a small model, a Core ML load and another macOS build would show how general it is; that is a precondition for reporting it upstream.
