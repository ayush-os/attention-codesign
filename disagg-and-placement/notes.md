# Disaggregated Serving + Placement — Working Notes

Live Prediction/Log-style working document for [`spec.md`](spec.md), kept
current during actual derivation — mirrors how `prefill_notes.md` and
`decode_notes.md` each describe superseding an original `notes.md` log once
their phase was done. Nothing here is polished prose yet; this is the record,
not the deliverable.

---

## Phase 0 — Setup

### Reference reading (🔧)

**[vLLM / PagedAttention](https://arxiv.org/abs/2309.06180)** (paper +
[real v1 source](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py)):

- KV cache divided into fixed-size blocks of **16 tokens** each; per-request
  **block table** maps logical block index → physical block (non-contiguous
  physical layout allowed) — literal OS virtual-memory paging applied to KV
  cache.
- Reported: **2–4× throughput**, **~4× effective batch size**, internal
  fragmentation cut from 60–80% (naive contiguous allocation) to ~10–15%.
- Confirmed directly in the real v1 source: `BlockPool` keeps a doubly-linked
  free-list ordered for **LRU eviction**; blocks are content-hashed for
  **prefix-cache reuse** across requests (copy-on-write on divergence); a
  **watermark** reserves a slice of free blocks specifically so waiting/
  preempted requests don't trigger a preemption cascade.
- Three responses to memory pressure: recompute (beam search only), swap to
  CPU, or preempt the request outright.

**[DistServe](https://arxiv.org/abs/2401.09670)**:

- **Interference mechanism** (why colocating prefill+decode hurts): a single
  prefill request injected into a decode-only batch inflates batch time
  **60ms → 200ms** (13B model, A100, input length 1024) — decode steps behind
  it in the batch get delayed.
- **Prefill:decode ratio**: no closed form — each phase's own goodput
  (req/s meeting its SLO) is profiled independently, then an allocation is
  solved via simulation (their Algorithm 1). Directionally: decode
  under-utilizes GPU compute, so more prefill instances per decode instance is
  normal.
- **KV handoff**: pull model — decode instances fetch from the prefill
  instance's own GPU memory as needed, using that memory as a queuing buffer.
  Cost is negligible: ~90 Gbps needed at 10 req/s (OPT-175B) vs. 600 GB/s
  NVLink intra-node / up to 800 Gbps Infiniband inter-node — **<0.1% of total
  latency**, 95% of transfers complete in <30ms.
- **Reported gains** vs. colocated vLLM: 2.0–3.41× more req/s (chatbot,
  ShareGPT), up to **4.48×** (summarization/long-context, LongBench), >90%
  SLO attainment throughout.
- **Workload model**: Poisson arrivals (1–4 req/s tested); ShareGPT-based
  chatbot benchmark run at **input≈512 / output≈64** tokens as representative
  values (their own synthetic test setup, not the raw dataset's own stats —
  see discrepancy note below).

**[Mooncake](https://arxiv.org/abs/2407.00079)**:

- **Three pools**: prefill pool, decode pool, and a separately-scaled
  **KVCache pool** spanning GPU/CPU-DRAM/SSD, connected via RDMA (800 Gbps,
  async, overlapped with compute), orchestrated by a global scheduler
  ("Conductor").
- **Why a separate pool, not DistServe's pull-from-prefill-GPU model**: doing
  this decouples cache retention from compute-memory pressure. It's actually
  two distinct jobs sharing one mechanism: (a) **handoff buffer** for KV
  cache that's finished prefill but not yet claimed by a decode machine
  (DistServe's role, but off the compute GPU entirely), and (b) **persistent
  cache** for reuse across unrelated requests later (shared system prompts,
  multi-turn history) — real value only if retention isn't capped by
  whatever HBM a prefill GPU happens to have free.
- **Placement**: hot KV blocks replicated across prefill nodes to avoid fetch
  congestion; cold blocks pushed to CPU/SSD. Reuse-vs-recompute decision is
  threshold-based (remote prefix match length vs. local reusable length × a
  threshold), not "always reuse if possible."
- **Reported gains**: up to **525%** throughput at long context (128k tokens)
  vs. colocated vLLM; **75% more requests** at matched SLOs on a real
  23k-request trace (vLLM met its TBT SLO only 57% of the time vs. ~100% for
  Mooncake).

**Checkpoint numbers to sanity-check the eventual model against (Phase 4)**:
DistServe's throughput multipliers by workload class, its <0.1%-of-latency
KV-transfer claim, Mooncake's SLO-attainment gap and 525%/75% headline
figures, PagedAttention's fragmentation/batch-size numbers.

---

### The 🧠 twist: what the model needs to represent

**Simulation style — discrete-event, not closed-form.** Unlike the attention
project's static roofline math (fixed workload shape, no notion of time) or
even the MoE project's closed-form comms-volume derivation, this project's
core questions — queue wait time, hardware utilization, chip-ratio balance,
memory-pressure-driven eviction — only exist once requests arrive over time
and contend for shared resources. Mirrors why the MoE project reached for
ASTRA-sim (a real event-stepped tool) once contention/ordering became the
question, just hand-built here since no off-the-shelf tool fits (per spec
Phase 0).

**Entities**: prefill machines, decode machines, one finite intermediate KV
pool.

**Request lifecycle (state machine)**:

```
arrive → [queue: wait for free prefill machine]
       → prefill runs (duration TBD, Phase 1/2 — tie to attention project's
         FLOPs/roofline numbers)
       → KV cache ships to intermediate pool; prefill machine frees
         IMMEDIATELY (Mooncake-style decoupling — not DistServe's pull model,
         where the KV cache sits in the prefill GPU's own memory as the
         buffer, coupling prefill throughput to decode availability)
       → [queue: wait for decode capacity in the pool]
       → transported to a decode machine
       → joins as one of up to N concurrently active requests on that
         machine, bounded by remaining HBM after weights + other active
         requests' KV cache
       → steps until generation finishes
       → frees its decode slot, request complete
```

**Decode granularity — the real fork, resolved to a deliberate middle
ground.** Neither of the two extremes:
- *Full Orca/vLLM-style continuous batching* (iteration-level scheduling,
  chunked-prefill injection mid-batch, preemption priority policies) — real
  complexity that doesn't serve this project's actual questions (chip ratio,
  handoff cost, placement policy), the same call the attention project made
  when it stopped at RTL/Verilator instead of chasing synthesis-level
  area cost.
- *One request per decode machine at a time* — would mean a decode machine
  never holds more than one request's KV cache simultaneously, which guts
  Phase 1's "how many tokens of KV cache fit in HBM" question and Phase 3's
  eviction-under-pressure question outright: there'd never be multiple KV
  caches actually contending for the same memory.

**Resolved to**: N concurrently active requests per decode machine, capacity-
bounded by memory (weights + Σ active requests' KV cache ≤ HBM). Justified
directly by the attention project's own finding: decode sits **~30–240×
below** the compute ridge point *per single request* — a lone decode request
massively underutilizes a machine, the same underutilization DistServe itself
cites as the reason to run multiple things concurrently on the decode side.

**Intermediate KV pool — finite capacity, decided over unbounded.** An
unbounded pool would eliminate the memory-pressure question this whole
project exists to study (Phase 1's "how much fits" and Phase 3's "what gets
evicted" both go vacuous). **Placeholder policy for pool-full**: block until
space frees — simplest structurally-correct option (nothing silently
disappears or silently succeeds when it shouldn't), deliberately *not*
solving real eviction policy yet. Mirrors Phase 1b's "flag, don't force"
pattern (accumulator capacity left as an open Timeloop sweep parameter) and
the MoE project's punt on the α1/α2/α3 balance-loss coefficients — real
eviction policy is explicitly Phase 3's job, and solving it before the sim
can even show whether/how often the pool fills up would be optimizing blind.

**Arrival process — Poisson, rate λ as the sweep parameter.** Matches
DistServe's own arrival model. Chosen over fixed/deterministic spacing
(understates queueing — queues build from burstiness, not average rate) and
over trace-driven replay (no real production trace on hand). λ is also
exactly the sweep axis needed for the experiment this project wants to run:
"how does wait time / utilization respond as request volume changes."

**Request shape (prompt/generation length) — lognormal, anchored to
DistServe's own benchmark numbers.** Real ShareGPT corpus stats are
right-skewed (mean ≈56 tokens, median ≈21 — [source](https://arxiv.org/pdf/2604.00499)-adjacent
search, not DistServe itself), but DistServe's own reported ShareGPT-based
chatbot benchmark used **input≈512 / output≈64** as representative values.
**Decision: anchor to DistServe's numbers, not the raw corpus stats** — since
Phase 4 needs comparability against DistServe's own reported throughput/
latency figures, matching their benchmark's regime matters more than matching
the raw dataset. Lognormal (not fixed-length) specifically because length
variance, like arrival burstiness, is part of what makes queueing/memory-
pressure behavior realistic. Flagged explicitly as a low-priority
approximation — not load-bearing to get exactly right.

---

### Chip choice — TPU 8i, homogeneous (prefill *and* decode)

**Final: TPU 8i for both pools.** Arrived at after first choosing, then
deliberately reversing, a heterogeneous TPU 8i (prefill) / Groq (decode)
split — full reasoning on both sides kept below rather than scrubbed, per
this project's own house style of keeping rejected paths on record.

**Why heterogeneous Groq-decode was tried first**: on TPU 8i alone, the
Phase 1 capacity math would just repeat the MoE project's own weights/HBM-
budget derivation with different totals — no new question. Groq is
architecturally SRAM-only (no HBM at all), which is the exact "SRAM-only /
no-HBM architectures (Groq, Cerebras precedent)" thread `decode_notes.md`
flagged and explicitly deferred ("would need its own ridge-point
recomputation under SRAM-scale bandwidth") rather than chased. Decode sitting
~30–240× below the HBM ridge point *is* the argument for going there —
removing HBM from the roofline entirely, rather than tiling around it the
way Gemmini had to.

**Why it was reversed**: attempting the actual Phase 1 arithmetic surfaced a
problem no amount of "just do more math" fixes. Groq's per-chip SRAM (500MB)
can't hold Llama-3-70B's weights (≈70GB at FP8) at all — a real deployment
needs the model pipeline-sharded across ~140+ chips. That's not just harder
arithmetic than TPU 8i's "everything fits on one chip" case — it breaks the
unit of abstraction Phase 0's simulator design already committed to, where
"a decode machine" is one atomic unit holding N concurrent requests bounded
by *its own* memory. On Groq, "one decode machine" is actually a pipeline of
~140+ chips acting together, which would force reworking what "a machine"
means in the sim already built, not just adding a harder capacity
calculation on top. At that point the project stops being about
disaggregation/placement and starts also being about pipelined
model-parallelism on a memory-starved chip — precisely the scope-creep
failure mode `spec.md`'s own "Note on scope" section warns against ("a
single project trying to simultaneously handle sharding + disaggregation +
MoE + KV placement would force shallow treatment of each piece"). Decided
this is a legitimate, real future thread (a focused fourth project could do
it justice) rather than something to smuggle into this one — same "flagged,
not chased" move `decode_notes.md` already made with this exact idea once.

Considered and rejected (independent of the Groq reversal): Cerebras (a
wafer *is* the whole device — breaks the "chip ratio" framing this project
and the MoE project both depend on); Trainium/AMD/MTIA/Maia (all still
HBM-based, same regime as TPU, no new architectural axis).

**Groq reference chip specs — kept for a possible future project, sourced
but never used in any derivation here:**

*Groq "3 LPX" (current generation, chosen as canonical) — confirmed directly
via [groq.com](https://groq.com/lpu-architecture), stated at rack
granularity; per-chip figures below are exact division (256 chips/rack), not
estimates, cross-checked against secondary per-chip reporting (~500MB /
~150TB/s) for consistency:*

| Spec | Value |
|---|---|
| Rack | 256 LPUs, 128 GB aggregate on-chip SRAM, 40 PB/s aggregate SRAM bandwidth, 315 PFLOPS FP8 |
| Per-chip (implied) | 500 MB SRAM, ~156 TB/s bandwidth, ~1.23 PFLOPS FP8 |
| RealScale (per chip, secondary-sourced only) | 96 links × 112 Gbps ≈ 2.5 TB/s aggregate bidirectional |
| Real-world precedent | 256-chip rack deployed for **Llama-3.3-70B at FP8** — same model family this project already uses |

*Gen-1 LPU (superseded, kept as a fallback/cross-check — far more
independently corroborated across sources than "3 LPX," but an older
generation with an older precedent workload):*

| Spec | Value |
|---|---|
| Per-chip | 230 MB SRAM, 80 TB/s bandwidth, 750 TOPS INT8 / 188 TFLOPS FP16, 14nm, 900MHz |
| RealScale (per card) | 11 links × 30Gbps × 4 lanes ≈ 330 GB/s aggregate bidirectional |
| Real-world precedent | ~576 chips (9 racks) for Llama-2-70B ([source](https://x.com/swyx/status/1759759125314146699)) |

**Superseded consequence** (applied only to the rejected heterogeneous
design, kept for the record): prefill (TPU 8i, Boardfly mesh) and decode
(Groq, RealScale) would have sat on two different-vendor interconnect
fabrics, forcing the prefill→decode KV handoff to cross between them — a
concrete instance of spec Phase 2's "same fabric or dedicated?" question.
Moot now that both pools are TPU 8i on the same Boardfly fabric throughout.

### Precision — uniform FP4 throughout

Reverted along with the chip decision — FP4/FP8 mixed precision (below,
kept for the record) only existed to match Groq's own grounded precedent,
which no longer applies once decode is TPU 8i too.

**Final: FP4 uniformly, both pools.** Same precedent the MoE project already
established for TPU 8i specifically (its native format) — now applies
end-to-end since there's only one chip type in the system. No requantization
mechanism at the prefill→decode handoff; KV cache moves at a consistent
precision throughout.

**Superseded reasoning** (applied only to the rejected heterogeneous design,
kept for the record): the attention project used int8 uniformly
(chip-agnostic), the MoE project used FP4 specifically because it's TPU 8i's
native format — neither matched a Groq decode pool cleanly, which motivated
mixed FP4 (TPU)/FP8 (Groq) and a real requantization-at-handoff mechanism
(the same pattern as Phase 1a's P-matrix int8 requantization, just at an
inter-chip boundary instead of an intra-chip HBM round-trip). Moot with
homogeneous TPU 8i.

---

### Open threads carried forward from Phase 0

- **Real eviction/placement policy for the intermediate pool** — deferred to
  Phase 3 by design; only a "block until space frees" placeholder exists now.
- **Interconnect fabric for prefill→pool→decode transfer**: same Boardfly
  fabric the MoE project already uses, or a logically dedicated channel on
  the same physical network? Spec Phase 2's question — simpler than the
  cross-vendor version this was under the (rejected) Groq design, but still
  open.
- **Discrepancy** between raw ShareGPT corpus length stats (mean 56 / median
  21) and DistServe's own benchmark numbers (512 / 64) — noted, not
  reconciled; using DistServe's numbers for Phase 4 comparability.
- **Prefill/decode step duration formulas** — not yet derived; needs the
  attention project's FLOPs/roofline numbers plugged in (Phase 1/2).
- **SRAM-only decode (Groq/Cerebras), take two** — tried and deliberately
  reversed here (see Chip choice above) because it forces reworking Phase
  0's "machine" abstraction into a multi-chip pipeline, which is out of
  scope for a disaggregation/placement-focused project. Real, legitimate
  material for a focused future project of its own — not lost, just correctly
  not-this-project, the second time this exact idea has been flagged and
  deliberately deferred rather than chased (first in `decode_notes.md`).

---

## Phase 1 — Characterize the memory hierarchy problem

### 1.1 Weight footprint and remaining HBM

Llama-3-70B, FP4 (0.5 bytes/param): weights = `70×10⁹ × 0.5 = 35×10⁹ bytes`
= **35 GB**. TPU 8i HBM/chip = 288 GB (reused directly from the MoE
project). Remaining after weights = `288 − 35` = **253 GB**.

No sharding needed — unlike the rejected Groq design, weights comfortably
fit on a single chip, confirming the homogeneous-TPU-8i call was the right
practical tradeoff (see Chip choice, above).

### 1.2 KV cache bytes/token — the growth curve

Formula (GQA, summed across all layers):
`bytes/token = 2 (K+V) × precision_bytes × d_head × n_kv_heads × n_layers`.

At FP4 (0.5 B), `d_head=128`, `n_kv_heads=8`, `n_layers=80` (verified via
[HF config](https://huggingface.co/unsloth/Llama-3.3-70B-Instruct/blob/main/config.json)
— a genuinely new number for this project; the attention project only ever
analyzed one layer in isolation and never needed a total layer count):

`2 × 0.5 × 128 × 8 × 80 = 81,920 bytes/token (80 KiB/token)`.

This *is* spec Phase 1's requested growth curve: `KV(context_len) =
context_len × 81,920 bytes`, linear, GQA already folded in via
`n_kv_heads=8` (MHA would give 8× this, `n_heads=64`).

### 1.3 Activation/overhead reservation — mostly ruled out, narrowed to one real term

Worked through and **rejected** as material to the HBM reservation:
transient S/P/softmax-state activations, under the fused execution model
this project's own attention work already established (`prefill_notes.md`
§1.2/§2.3), live in TPU 8i's on-chip SRAM (384MB) during compute, not HBM —
a separate physical budget, following directly from "stationary is scoped to
a specific memory boundary" (`prefill_notes.md` §2.6, Key Takeaways #4/#8).
Q/O — the terms that genuinely do touch HBM — are real but negligible at
this scale (prefill's fused MHA Q+O ≈ 4 GiB at int8 per `decode_notes.md`
§1.4; decode's ≈ negligible; both dwarfed by KV cache's hundred-GB scale).

What's left, and real: **block-allocation fragmentation**, sourced directly
from this project's own Phase 0 reading rather than invented — PagedAttention's
reported 10–15% fragmentation under 16-token block allocation (vs. 60–80%
naive contiguous allocation). **Decided: reserve 10–15%** of the post-weights
HBM budget for this — not the MoE project's 46% margin, since that covered a
different mechanism (activation/router memory on a different chip) no
longer analogous once activations were ruled out here.

### 1.4 Aggregate KV-cache token capacity

`usable KV budget = 253 GB × (1 − reserve)`

| Reserve | Usable KV budget | Aggregate tokens |
|---|---|---|
| 10% | 227.7 GB | 227.7×10⁹ / 81,920 ≈ **2,779,541** |
| 15% | 215.05 GB | 215.05×10⁹ / 81,920 ≈ **2,625,122** |

**Range: ≈2.63M–2.78M aggregate tokens** of KV-cache capacity per TPU 8i
chip, at FP4, after weights and fragmentation overhead. This is an
*aggregate* figure — capacity shared across however many requests are
concurrently resident on one decode machine at once (Phase 0's "N
concurrently active requests bounded by memory" design), not a
single-request maximum.

### 1.5 Aggregate tokens → N concurrent requests (resolved)

Chose (a) from the fork below: a **hard per-request context-length cap of
8,192 tokens** — not arbitrary, it's Llama 3's own native context length,
and the exact `seq_len` already reused across every document in this repo
("for cross-project comparability," per the MoE project's own stated
reasoning for the same number) — reused a third time rather than
introducing a new ungrounded figure.

`N = budget / cap`:

| Reserve | Aggregate tokens | N (max-length requests) |
|---|---|---|
| 10% | 2,779,541 | `2,779,541 / 8,192` = **339** (339×8,192=2,777,088 fits; 340 wouldn't) |
| 15% | 2,625,122 | `2,625,122 / 8,192` = **320** (320×8,192=2,621,440 fits; 321 wouldn't) |

**N = 320–339 concurrent max-length requests per decode chip**, safe by
construction — the literal worst case (every slot simultaneously at the
8,192-token cap) still fits exactly.

**Follow-on, quantifying the cost of that conservatism**: N above assumes
every concurrent request is maxed out. Phase 0's own request-length model
(lognormal, ~576 tokens average — 512 in + 64 out, DistServe-anchored) says
most won't be. At the *average* length instead of the cap, the same budget
supports `2,779,541 / 576 ≈ 4,827` (10%) to `2,625,122 / 576 ≈ 4,558` (15%)
concurrent requests — **~14× more** than the worst-case-safe static number.
Doesn't resolve the hard-cap-vs-dynamic-admission fork below, but quantifies
how much throughput is actually at stake in that choice.

**Still open, still explicitly Phase 3's job**: (a) vs. (b) below wasn't
chosen for *production* semantics, just to get a concrete N for this
project's own sizing — either a hard per-request cap (chosen above, simple,
safe, ~14× conservative) or usage-tracked dynamic admission (mirrors the
intermediate pool's own "block until space frees" placeholder, recurring one
level down inside a single decode machine, real throughput but real
complexity). **What happens at the cap itself** (hard stop, matching real
chatbot behavior, vs. compaction/sliding-window, matching coding-agent-style
context management) is explicitly Phase 3 territory (spec's own pointer to
"attention sink / sliding window approaches"), not resolved here.

### 1.6 Core tension — static/fixed weights vs. dynamic/open-ended KV cache

Spec Phase 1's third ask, stated explicitly rather than left implicit.

**Real-world grounding first**: no real deployment replicates weights
anymore, especially at multi-trillion-parameter scale — sharding
(tensor/pipeline parallelism) amortizes the weight cost across a pool
instead of paying it again on every chip, a strictly better use of space
once a model is large enough. This project's homogeneous TPU 8i design
*does* replicate (every chip holds a full 35GB copy) — a deliberate
simplification (see Chip choice, above), specifically defensible **only
because of Llama-3-70B's scale relative to TPU 8i's HBM**: `35GB/288GB ≈
12%` wasted per chip is cheap enough to not matter. This would not hold at
trillion-parameter scale, where the model could exceed a single chip's
*entire* HBM, not just its SRAM — at that point sharding stops being merely
better and becomes mandatory, independent of memory type. Stated as an
explicit boundary condition on this project's simplification, not left
implicit.

**The durable tension, independent of replication vs. sharding**: weights
are a fixed-size cost, known the instant a model is chosen, paid once and
never revisited regardless of serving load. KV cache is open-ended — it
grows with every token generated for every live conversation, with no
natural ceiling except whatever hardware capacity or an explicit policy
imposes (§1.5's cap question).

**Implication for placement (Phase 3)**: weights are never a lever in the
placement story — they can't shrink, can't be evicted, can't be delayed.
**KV cache is the only elastic resource in the entire system** — every real
placement/eviction question this project asks from here on is about KV
cache, never about weights.

---

## Phase 2 — Disaggregation hypothesis (chip ratio)

### 2.1 Reframing: service time, not the compute/memory-bound label itself

Initial instinct was to derive the chip ratio directly from the prefill/decode
roofline *labels* (compute-bound vs. memory-bound) carried over from the
attention project. Caught before computing anything: the label alone doesn't
determine a throughput ratio — DistServe's own approach (Phase 0 reading)
profiles each phase's actual goodput independently and solves an allocation,
no closed form from the regime label. What roofline *does* supply is the
ingredient that turns FLOPs/bytes into **service time**: `time =
max(FLOPs/peak_compute, bytes/peak_bandwidth)`. Regime is real but one level
removed — the actual lever is service time → throughput per chip → chip
ratio.

### 2.2 Attention service time — reused from the sibling project, converted to this project's chip/precision

Reused `prefill_notes.md`/`decode_notes.md`'s own FLOPs and bytes formulas
(GQA, the real Llama-3-70B config used throughout this repo) rather than
re-deriving from scratch. Converted from their int8/TPU v5e basis to this
project's FP4/TPU 8i basis: FLOPs are precision-invariant (same MAC count);
bytes at FP4 = int8 bytes ÷ 2. TPU 8i FP4 specs used throughout: peak =
10.1×10¹⁵ FLOPs/s, HBM BW = 8.6×10¹² B/s (ridge point = 1,174.4 FLOPs/byte).

**Batch scaling — confirmed exact, not assumed**: `decode_notes.md` §1.2
states explicitly that batch is inert for SDPA's own AI ("every batch element
carries a distinct KV cache — no cross-batch reuse — so FLOPs and bytes
scale together, linearly, with batch"). Checked the consequence directly:
since `service_time ∝ batch` and `throughput = batch/service_time`,
**attention's aggregate throughput per chip is invariant to batch size** —
the batch/N question that Phase 1 left open (320 vs. 339 vs. ~4,558) turned
out to be moot for attention's own throughput number. (Turned out **not** to
hold for FFN — see §2.5.)

### 2.3 Two real-workload corrections to the reused numbers

Both `prefill_notes.md`/`decode_notes.md` numbers were derived for the
attention project's own fixed workload (`seq_len=8192` throughout) — not
automatically valid for this project's own request-shape model (Phase 0:
DistServe-anchored, ~512 avg prompt / ~64 avg output tokens). Two corrections
made, not silently:

- **Prefill seq_len: 512, not 8192.** Prefill FLOPs scale as `seq_len²`;
  bytes scale as `seq_len` — so this isn't a minor rescale (`(512/8192)² =
  1/256` on FLOPs alone). Recomputed at seq_len=512, GQA, batch=32, fused,
  FP4: **FLOPs = 2³⁸ ≈ 2.749×10¹¹, bytes = 150,994,944 B**. Real finding: at
  this realistic prompt length, prefill is only **marginally compute-bound
  (~1.55× margin)** — a much closer call than the confidently-compute-bound
  picture (~17–25×) the sibling project found at seq_len=8192. AI scales
  linearly with seq_len, so shrinking the prompt shrinks the margin, not just
  the absolute time.
- **Decode context length: 544, not 8192.** `seq_len_kv=8192` in the sibling
  project was Llama-3's native max-context cap, not an average. Since
  decode's context grows by 1 token/step across the generation (512 → 576,
  using Phase 0's averages), and bytes/step scale **linearly** in context
  length, the time-averaged context across a request's decode phase is
  exactly `avg_prompt_len + avg_output_len/2 = 512 + 32 = 544` (arithmetic
  mean of a linear trajectory's endpoints). At FP4, batch=32, seq_len_kv=544,
  GQA: **FLOPs = 570,425,344, bytes = 18,087,936 B**. *Flagged, not fully
  resolved*: this uses the lognormal request-shape model's mean as a point
  estimate, not the full distribution — since throughput is inversely
  proportional to context length, the true distributional average
  (harmonic-mean-flavored) isn't identical to throughput-at-the-mean. Treated
  as a defensible approximation, consistent with this project's existing
  precedent of flagging rather than fully resolving the lognormal choice
  itself.

**N (concurrent decode requests) — two decisions, different weight given to
each:**
- 320 vs. 339 (Phase 1's 15%/10% fragmentation-reserve split, only a ~6%
  spread): judged as noise, picked **N=320**.
- 320 vs. **~4,558** (hard-cap vs. average-case dynamic admission, a **~14×
  spread** — Phase 1 §1.5's real open fork): judged load-bearing enough to
  carry both forward as scenarios rather than collapsing now, mirroring this
  repo's own precedent (computing both fused/unfused, both MHA/GQA, rather
  than picking one early). **Only N=320 has actually been run through the
  ratio below — the ~4,558 scenario is still open, not yet computed.**

### 2.4 The FFN gap — a first-order omission, not a rounding error

Both `prefill_notes.md`/`decode_notes.md` are explicitly SDPA-only (§0 of
each: "not the surrounding QKVO/FFN projection weights"). Neither sibling
project ever derived FFN compute for Llama-3-70B — the MoE project's own FFN
formula is for **DeepSeek-V2** (sparse MoE, `d_model=5,120`/`d_ff=1,536`, only
8-of-162 experts active/token) — architecturally wrong model, wrong dims,
wrong sparsity for Llama-3-70B (dense, every token through one full
FFN/layer). The formula shape (`3×2×d_model×d_ff` per token, SwiGLU)
transfers; the dims don't.

Sourced Llama-3-70B's real FFN dims directly (matching this project's own
practice of pulling `n_layers=80` from HF config rather than assuming):
**`d_ff` (intermediate_size) = 28,672**, `hidden_act = silu` — confirming
SwiGLU is the real architecture, not a simplifying assumption, via
`unsloth/Llama-3.3-70B-Instruct`'s config (WebFetch), independently
cross-confirmed against a *How to Scale Your Model*-style hyperparameter
table the user supplied directly (identical values: `n_layers=80,
d_model=8192, d_ff=28672, n_heads=64, n_kv_heads=8, d_head=128`; also
surfaced `n_embeddings (vocab) = 128,256`, uncounted anywhere — flagged as an
open gap, see §2.7).

**FFN per token/layer**: FLOPs = `3×2×8,192×28,672` = **1,409,286,144**.
Weight bytes/layer (FP4, weight-stationary/fused — same idealization
attention used for K/V) = `3×8,192×28,672×0.5` = **352,321,536 B**,
batch-invariant (loaded once, amortized across every token in the batch —
the mechanism that turned out to matter, §2.5).

### 2.5 First combined result (batch=32 throughout) — looked backwards, investigated, resolved

**Combined (attention+FFN) per-layer service time, batch=32 both pools:**

| | Prefill (seq_len=512) | Decode (seq_len_kv=544) |
|---|---|---|
| Attention service | 27.22 µs (CB, 1.55×) | 2.103 µs (MB, 37.2×) |
| FFN service | 2,286.1 µs (CB, 40.4×) | 41.00 µs (MB, 9.18×) |
| Combined/layer | 2,313.3 µs | 43.10 µs |
| × 80 layers | **185.07 ms** | **3.448 ms** |
| Throughput/chip | 32/0.18507s ≈ **172.9 req/s** | (32/0.0034483s)/64 ≈ **145.0 req/s** |

FFN dominates both phases (~99% prefill, ~95% decode) — attention alone was a
rounding error, explaining why the earlier attention-only ratio (0.0136, ~73
decode chips per prefill chip) was so obviously broken.

**Ratio at batch=32: `decode_tput/prefill_tput = 145.0/172.9 ≈ 0.84`** — i.e.
needing *more* decode chips than prefill chips. Backwards from DistServe's
own reported direction ("more prefill instances per decode instance is
normal"). Investigated rather than accepted or hand-waved as "different
model."

**Root cause, confirmed mechanistically**: FFN weight bytes are
batch-invariant (§2.4) — using batch=32 for decode's FFN badly
under-amortized the fixed 352 MB weight load, when this project's own Phase 1
had already derived a realistic decode concurrency ceiling an order of
magnitude higher (N≈320–339). Attention has no such batching payoff (bytes
scale linearly with batch, confirmed §2.2) — so the batch=32 default, correct
for attention, was silently wrong for FFN.

**Recomputed decode at N=320:**

| | Attention (N=320) | FFN (N=320) |
|---|---|---|
| FLOPs | 5,704,253,440 | 450,971,566,080 |
| Bytes (FP4) | 180,879,360 | 354,942,976 |
| Compute time | 0.5648 µs | 44.65 µs |
| Memory time | 21.03 µs | 41.27 µs |
| Service | 21.03 µs (MB, 37.2× — unchanged, confirms attention's batch-invariant AI) | **44.65 µs (CB, ~1.08× — crosses regime)** |

FFN's decode regime **flips from memory-bound (9.2× at N=32) to marginally
compute-bound (~1.08× at N=320)** — exact crossover solved at **N≈296**,
landing right inside the 320–339 range already chosen for unrelated reasons.
Confirmed N=320 vs. 339 still doesn't matter post-FFN (token throughput
60,904 vs. 60,900 tok/s — effectively identical, both past the crossover).

**Corrected decode**: combined/layer = 21.03+44.65 = 65.68 µs; ×80 layers =
**5.254 ms**; token throughput = 320/5.254ms ≈ 60,904 tok/s; request
throughput (÷64) ≈ **951.6 req/s/chip** — a **6.6× jump** from the batch=32
figure.

**Corrected ratio: `951.6/172.9 ≈ 5.50`** → **~5.5 prefill chips per decode
chip** — matches DistServe's directional finding.

**Ruled out, explicitly**: the "different model" hypothesis (Llama-3-70B/GQA
vs. DistServe's OPT-175B/plain-MHA) as the primary explanation. Checked
directly: GQA's ~8× K/V fetch reduction would, if anything, make *this*
project's decode cheaper/faster relative to an OPT-175B comparison — pushing
toward needing *fewer* decode chips, i.e. the same direction as the
already-backwards batch=32 result, not toward resolving it. The
batch-size/FFN-amortization gap is the real, load-bearing explanation — a
genuine correctness catch in this project's own numbers, not an
apples-to-oranges artifact.

### 2.6 Final chip-ratio hypothesis (as of this phase)

**~5.5 prefill chips per decode chip**, derived from attention+FFN combined
service time on TPU 8i/FP4, at this project's own real workload parameters
(seq_len=512 prompt, seq_len_kv=544 average context, N=320 decode
concurrency, DistServe-anchored 512-in/64-out request shape). Matches
DistServe's own reported direction (more prefill instances needed) after the
FFN correction — not a forced match, a mechanistically-explained convergence.

### 2.7 Key Findings — Phase 2 (chip ratio)

1. **Regime label ≠ ratio driver.** Compute/memory-bound tells you which term
   sets service time; the ratio itself needs service time → throughput →
   chip count, exactly DistServe's own no-closed-form approach.
2. **Batch invariance is lever-specific, not universal.** Attention's AI is
   batch-invariant (no cross-request reuse); FFN's is not (weight bytes are
   batch-invariant, which is the opposite property, and the one that
   actually matters for chip provisioning).
3. **Reused numbers need reused-workload verification, not just
   reused-formula verification.** Two real mismatches surfaced (prefill
   seq_len, decode context length) purely from checking whether the sibling
   projects' *fixed workload* matched this project's own *request-shape
   model* — same category of gap both times.
4. **The FFN omission was the dominant source of the "backwards" ratio, not
   model choice.** Attention alone was <1–5% of either phase's real service
   time — validating attention-only would have validated the wrong thing.
5. **A "boring" match with DistServe's direction after a real correction is a
   stronger result than an unexplained match would have been** — the
   ~5.5:1 ratio wasn't tuned to match DistServe; it fell out of fixing a
   genuine mechanistic gap (FFN batch-amortization), and happened to land in
   the expected direction.
6. **"Decode is hard" and "prefill needs more chips" are not in tension —
   they're the same fact from two different angles.** "Decode is hard" is a
   per-request compute-*utilization* statement (memory-bound → the compute
   engine sits mostly idle). The chip-ratio question is about aggregate
   *throughput per chip*, and those pull in opposite directions precisely
   because of *why* decode is memory-bound: at batch=32, decode's FFN margin
   was 9.2× memory-bound — meaning ~9× idle compute headroom, directly
   exploitable by batching since FFN's weight bytes are batch-invariant while
   FLOPs aren't (§2.4/§2.5). That idle headroom is exactly what N=320
   converted into the 6.6× throughput jump. Prefill's FFN margin at batch=32
   was already 40.4× *compute*-bound (memory time ~2.5% of total) — there's
   no comparable idle capacity left to exploit via batching, so its
   throughput per chip is close to fixed and adding capacity means adding
   chips. **General lesson: the more severely memory-bound a workload is, the
   more upside batching creates, because there's more idle compute to
   convert — being "hard" per-request and being "cheap to scale" via batching
   are the same underlying property, not opposing ones.**

### 2.8 Open threads carried forward

- **KV-handoff mechanism** (spec's other Phase 2 ask — transfer cost,
  interconnect fabric) — not yet started. This section only covers the
  chip-ratio half of Phase 2.
- **N≈320 vs. N≈4,558 (hard-cap vs. average-case admission)** — only N=320
  has been run through the ratio; the ~14× scenario is still open, per §2.3.
- **Prefill's own batch=32 never reconsidered upward** — decode's N=320 came
  from Phase 1's derived HBM-capacity ceiling; no equivalent "prefill
  capacity" derivation exists in this project to justify raising prefill's
  batch the same way. Left at the attention project's inherited default, not
  because it's necessarily right, but because there's no derived alternative
  yet.
- **Vocab/LM-head matmul (`n_embeddings=128,256`)** — surfaced via the user's
  hyperparameter-table screenshot, never counted anywhere in either sibling
  project or here. Likely small relative to FFN (`d_model × vocab` once per
  token, not once per layer) but not verified.
- **FFN's "fused" SwiGLU intermediate treated as an assumption, not
  stress-tested** — mirrors attention's own fused/unfused split
  (`prefill_notes.md` §1.2), but unlike attention, never checked against
  SRAM capacity or an unfused alternative. If FFN's gate/up intermediate
  (`d_ff` elements/token) had to round-trip HBM instead of staying on-chip,
  this would materially change the FFN bytes-moved number.
- **Lognormal mean used as a point estimate for decode's average context
  length (544)** — flagged, not resolved via full distributional integration
  (§2.3).

### 2.9 QKVO correction — the SDPA-only gap fixed (supersedes §2.5/§2.6's ratio)

**Root cause** (full audit in §2b.11, done during the MoE leg but the fix
belongs here): `prefill_notes.md`/`decode_notes.md`'s attention numbers are
explicitly SDPA-only ("not the surrounding QKVO/FFN projection weights") —
correct for that project's own microarchitecture question, but silently
incomplete once reused here for throughput/service-time purposes. §2.4
caught and fixed the FFN half of the gap; QKVO itself was never added,
for either phase, until now.

**QKVO FLOPs/bytes per token/layer** (`d_model=8,192`, `n_heads=64`,
`d_head=128`, `n_kv_heads=8`, standard `2×M×N` per projection):
`Q_proj+O_proj = 2×2×d_model×(n_heads·d_head)`, `K_proj+V_proj =
2×2×d_model×(n_kv_heads·d_head)` → **301,989,888 FLOPs/token/layer**,
weight bytes (FP4) = **75,497,472 B** — exactly **21.4% of FFN's magnitude**
in both FLOPs and bytes (§2b.11). Weight-stationary/batch-invariant, same
amortization treatment as FFN's own bytes (§2.4).

**Prefill** (batch=32, seq_len=512 → 16,384 tokens/step): QKVO FLOPs =
16,384×301,989,888 ≈ 4.948×10¹²; compute_t=489.88µs, mem_t=8.78µs →
**compute-bound** (consistent with prefill's existing regime). QKVO service
= 489.88µs, added to the existing 2,313.3µs combined/layer →
**new combined/layer = 2,803.2µs**; ×80 layers = 224.26ms; **throughput =
142.70 req/s/chip** (down from 172.91, **−17.48%**).

**Decode** (N=320): QKVO FLOPs = 320×301,989,888 ≈ 9.664×10¹⁰;
compute_t=9.57µs, mem_t=8.78µs → **compute-bound** (a change from the rest
of decode's memory-bound FFN — QKVO's own regime, evaluated independently,
same as attention/FFN each get their own). QKVO service = 9.57µs, added to
the existing 65.68µs combined/layer → **new combined/layer = 75.25µs**;
×80 layers = 6.020ms; token throughput = 53,157.6 tok/s; **request
throughput = 830.59 req/s/chip** (down from 951.58, **−12.72%**).

**Corrected ratio**: `830.59/142.70 ≈ 5.82` → **~5.82 prefill chips per
decode chip** (up from §2.6's ~5.50, a **+5.8% shift** — real but bounded,
not order-of-magnitude). Mechanism: QKVO hits both phases, but prefill's
service time is more heavily weighted toward compute (already 40.4×
compute-bound before this fix) — adding another purely compute-bound term
lands proportionally harder there (+21.2% combined/layer time) than on
decode's already-mixed regime (+14.6%), which is why the ratio moves in the
direction of needing slightly *more* prefill chips per decode chip, not
fewer.

**This is now the authoritative dense chip-ratio figure (~5.82:1),
superseding §2.6's ~5.5:1** — kept on record above rather than edited, per
this project's own convention (Groq reversal, FP4/FP8 precision reversal).
The mechanistic story from §2.7's Key Findings (FFN batch-invariance
driving the whole result, the "decode is hard ⇒ cheap to scale" argument)
is unaffected — QKVO doesn't change *which* term dominates either regime,
only adds a modest, now-correctly-counted addition on top.

**Why the ratio barely moved despite a real 21.4%-of-FFN correction — an
Amdahl's Law mechanism, same shape `prefill_notes.md` already named once for
a different lever.** `impact_on_total = local_magnitude × share_of_total`:
QKVO's local magnitude is fixed (21.4% of FFN, always), but FFN's own share
of each phase's total differs (98.8% prefill, 68% decode) — so the same
local addition produces different total impact per phase (21.15% vs.
14.55%), not because the two hits were close to equal (they weren't — a
real ~45% relative difference) but because *neither* hit was large enough
on its own to swing a ratio (both phases retained >82% of their original
throughput). `prefill_notes.md`'s own Key Takeaway #3 named this exact
mechanism for GQA's byte savings (*"the real 8× local win produces ~0%
total win... purely a function of what fraction of the current bottleneck
it touches"*) — different lever (an added cost here, vs. a removed one
there), identical structural shape: a fixed local change, scaled by the
affected term's variable share of the total.

---

## Phase 2b — MoE chip ratio

### 2b.1 Per-expert selection-probability distribution — Zipf, calibrated to Gini≈0.70

Checked `moe-routing-notes.md` for a reusable per-expert selection-probability
model before inventing one (same discipline as reusing formulas elsewhere in
this project) — confirmed it doesn't have one. §2.2's Gini≈0.70/~3.3×
multiplier is a **device-level** load-imbalance magnitude, not a distribution
over the 162 individual experts; nothing there answers "expert *i* gets
picked with probability p_i." Genuinely new for this phase.

**Delegated choice** (user: pick one, not worth deliberating): **Zipf
distribution**, `p_i ∝ 1/i^s` over the 162 experts, exponent `s` solved
numerically so the resulting Gini coefficient matches ≈0.70.

**Why Zipf over a two-tier hot/cold split**: `moe-routing-notes.md` §2.2
already rejected a Pareto-by-analogy shape once for lack of MoE-specific
evidence. A bimodal hot/cold split has the identical problem — it requires
inventing an arbitrary hot-expert-count cutoff with nothing to source it.
Zipf is single-parameter, standard for modeling skewed discrete/categorical
selection frequency generally (default assumption for hot-key access
patterns in caching/DB literature), and calibrates smoothly to the one real
sourced number available (Gini≈0.70) without a second free parameter.

### 2b.2 Expected distinct experts touched as a function of batch size N

**Formula, exact given marginal probabilities**: `E[distinct] = 2 +
Σᵢ₌₁¹⁶⁰ [1 − (1−qᵢ)^N]`, where `qᵢ` = probability a single token touches
routed expert *i*, and the 2 shared experts are always touched
(deterministic, added outside the sum). Holds exactly regardless of the
M≤3-device-limited joint selection mechanics inside one token's own routing
— by linearity of expectation, expected distinct count only depends on each
expert's *marginal* per-token touch probability, not on how a token's 6
picks are jointly correlated with each other. Requires token routing
decisions independent *across* tokens (reasonable: unrelated requests).

**qᵢ, the one approximated piece**: `qᵢ = 1 − (1−pᵢ)⁶` — treats the 6 routed
slots as independent with-replacement draws from the Zipf weights `pᵢ`
(§2b.1), rather than modeling the real without-replacement top-6 selection
exactly (which has no clean closed form here — would need a Plackett-Luce-
style sequential-removal computation). **Flagged, not resolved**: with-
replacement duplicate draws can waste a slot re-picking an already-likely
expert, understating true single-token inclusion probability for the most
popular experts vs. real without-replacement selection — i.e. this
approximation is a conservative *underestimate* of coverage; the true curve
likely saturates even faster than reported below.

**Computed** (`s=1.076` from §2b.1, `n_routed=160`):

| N | E[distinct experts] | % of 162-expert table |
|---|---|---|
| 1 | 7.2 | 4.4% |
| 32 (dense's inherited default batch) | 66.4 | 41.0% |
| 100 | 114.3 | 70.5% |
| 296 (dense's own FFN crossover N, §2.5) | 151.9 | 93.8% |
| **320 (this project's own decode-N, §1.5)** | **153.5** | **94.7%** |
| 1,000 | 161.9 | 99.9% |
| ≥4,558 | 162.0 | 100.0% |

**Finding: coverage saturates fast, well before realistic decode
concurrency.** By N=320 — the same decode-concurrency figure already derived
in Phase 1 and reused throughout Phase 2a — the batch already touches 94.7%
of the full 162-expert table. Real amortization headroom exists at small N
(32 tokens touches only 41% of the table), but it's almost entirely spent by
the time batch size reaches this project's own realistic decode-N. This
points decisively toward spec_v2's **high-diversity limit**, not the
low-diversity one: at N=320+, expert-table cost is already close to its
ceiling, the opposite of dense's story (where more N kept buying more
amortization all the way to its own crossover at N≈296). Next step (spec_v2
item 2): convert this distinct-count curve into effective FFN bytes moved
per decode step as a function of N, to see how close "94.7% of the table"
actually lands to dense's flat 352 MB/step baseline.

### 2b.2b Deployment model — full replication, one TPU 8i chip, not expert-parallel sharding

**Question raised**: `moe-routing-notes.md` deploys DeepSeek-V2 expert-parallel —
160 routed experts sharded 20/device across an 8-device EP group, only 2
shared experts replicated (source: lines 58–67). Does this project have to
follow that, or is deployment model this project's own choice?

**Checked first, not assumed**: does the *full* (unsharded) model actually
fit on one TPU 8i chip? All 162 experts × 59 MoE layers + attention × 60
layers ≈ 234.46B params (matches `moe-routing-notes.md`'s own ≈234.5B
consistency check) ≈ **117.2 GB at FP4** — comfortably within TPU 8i's 288GB
HBM (≈41% used, 170.8GB free for KV cache). Unlike Groq (500MB SRAM vs. 70GB
Llama weights — categorically infeasible), there is **no memory-capacity
forcing function** requiring sharding here. Sharding would be a choice, not
a necessity.

**Three deployment options weighed**:
- **(A) Full replication, one atomic TPU 8i chip** — every prefill/decode
  chip holds the full 162-expert table locally, zero dispatch/combine
  traffic, routing is a pure local lookup.
- **(B) Expert-parallel sharding, 8-device EP group** (project #3's own
  model, as-is) — forces redefining "one decode machine" as an 8-chip group
  acting together, the same abstraction-breaking shape that got the
  Groq/heterogeneous design reversed in Phase 0. Requires reworking §2b.2 to
  a per-device (n=20 local experts) version with its own token-arrival model.
  The comms-cost question this reopens is already answered by project #3
  (decisively compute-bound, 5.0–6.7× margin, imbalance-proof) — re-deriving
  it here would mostly duplicate project #3's own work, not add new value.
- **(C) Abstract a whole rack (NVL72/Hopper-8) as one compute unit** —
  solves a problem that doesn't exist once (A) is confirmed feasible; also
  reopens the exact heterogeneous-chip comparability problem the Groq
  reversal avoided, and discards the TPU 8i/FP4/Boardfly continuity kept
  since Phase 0 for no offsetting benefit.

**Decided: (A), full replication on one atomic TPU 8i chip**, both pools.
Reasoning: verified feasible (unlike Groq); preserves Phase 0's atomic
decode-machine simulator abstraction exactly, so MoE plugs into the same
entity model as dense per spec_v2's own design intent; avoids duplicating
project #3's already-complete EP-sharding/comms analysis; keeps §2b.2's
global (n=162) distinct-experts-touched computation directly valid rather
than requiring a per-device rework. Sharding is real, legitimate material —
flagged as a future-project thread, the same "flagged, not chased" move
already made twice for SRAM-only/Groq decode (see Phase 0's Chip choice
section).

### 2b.3 Effective FFN bytes moved per decode step, as a function of N

`bytes/layer(N) = E[distinct(N)] × bytes_per_expert`, where
`bytes_per_expert = 3×d_model×d_ff×0.5 = 3×5,120×1,536×0.5 = 11,796,480 B ≈
11.80 MB` (FP4, SwiGLU gate/up/down — same per-expert formula project #3
already established, reused not re-derived).

| N | E[distinct] | MoE bytes/layer | vs. dense's flat 352.32 MB (§2.4) |
|---|---|---|---|
| 1 | 7.2 | 84.8 MB | 0.24× |
| **8.71** | — | **352.3 MB** | **crossover — MoE passes dense's flat number here** |
| 32 | 66.4 | 783.2 MB | 2.22× |
| 232 (§2b.2's 90%-of-table point) | 145.8 | 1,719.6 MB | 4.88× |
| **320 (this project's own decode-N)** | **153.5** | **1,810.6 MB** | **5.14×** |
| ≥4,558 (saturated) | 162.0 | 1,911.0 MB | **5.42× (ceiling)** |

**Finding, counter to the naive "sparse routing moves less data" intuition:
MoE moves *more* bytes/layer than dense at every realistic N, not less.**
Crossover happens almost immediately (N≈9 tokens) and plateaus at 5.42×
dense's flat baseline. Mechanism: each individual expert is much narrower
than dense's single FFN (`d_ff=1,536` vs. `28,672`, ~19× smaller per
matrix), but there are 162 of them — the aggregate table (162×11.8MB≈1.91GB)
dwarfs dense's single 352MB block by the same ~5.4× margin the saturated
row shows. Sparse routing's payoff is relative to *not sharding at all*
(touching 162 experts instead of some larger pool), never relative to
dense's single-FFN cost — DeepSeek-V2's total FFN parameter budget
(225.5B) is just much larger than Llama-3-70B's (56.4B) to begin with.

**Sets up spec_v2 item 3 (regime crossover)**: MoE decode's memory-time is
already far larger than dense's at any realistic N, while §2b.2's
distinct-experts curve saturates by N≈232 — very little room left for
further batching to buy a compute-bound crossover the way dense's did at
N≈296. Next: convert this bytes curve into memory time, alongside MoE
decode's own FLOPs (attention + FFN, following the 2.1–2.5 chain), to check
whether a crossover exists at all.

### 2b.3b Reading the 5.42× finding correctly — compute win intact, bandwidth win is deployment-dependent

**Question raised**: does §2b.3's finding (MoE moves *more* bytes/layer than
dense at every realistic N) mean MoE fails at its own stated purpose?

**No — the compute win and the memory-bandwidth win are separate claims, and
only one of them depends on the §2b.2b deployment choice.**

**Compute win, checked directly against Llama-3-70B, still fully intact**:
dense FFN FLOPs/token/layer = `3×2×8,192×28,672 = 1,409,286,144`; MoE FFN
FLOPs/token/layer (8 active experts) = `8×3×2×5,120×1,536 = 377,487,360` —
**MoE does ~3.73× fewer FLOPs/token/layer than dense**, despite DeepSeek-V2
having ~3.4× more total parameters (236B vs. 70B). FLOPs cost is
`N×8×FLOPs/expert` regardless of deployment model — this holds whether
weights are replicated or sharded, so nothing in §2b.3 threatens it.

**Memory-bandwidth win is real only under sharding, and full replication
(§2b.2b's choice) specifically forfeits it.** Sparsity only saves
weight-loading bytes if "the whole table" is something a device would
otherwise have to fetch. Under full replication every chip already holds
all 162 experts permanently — the only thing left to save is how much of
that already-resident table gets pulled off HBM per step, and §2b.3 showed
that saturates almost immediately (past dense's flat number by N≈9, past
90% of the table by N≈232). Under **sharding** (the deployment
`moe-routing-notes.md` actually uses, Option B from §2b.2b), a device only
ever loads from its own bounded local 20-expert shard — genuinely sparse,
never racing toward the global 162-expert ceiling. Sharding is the
mechanism that keeps MoE's bandwidth savings real at scale; full replication
trades that benefit away.

**Framing for the write-up**: §2b.3's 5.42× number is not evidence MoE
underperforms dense — it's a quantified cost of the Option A deployment
simplification chosen in §2b.2b (kept for simulator-abstraction reasons),
not a property of MoE sparsity itself. State both halves together in Phase
4/5, not just the bytes-moved number in isolation.

