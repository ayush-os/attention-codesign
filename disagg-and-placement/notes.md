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
