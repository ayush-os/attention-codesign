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
