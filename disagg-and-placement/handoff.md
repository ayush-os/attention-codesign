# Handoff — read this first in a new conversation

Written because the conversation that produced Phase 2c and Phase 3 got
long enough to need a fresh chat. This file's only job is to get a new
session oriented fast and working the same way the old one did. It is not
a project doc — nothing here should get cited from `notes.md`; if something
here turns out to matter long-term, it belongs in `notes.md`, not here.

## What this is

Third of three sequential codesign projects in this repo (see top-level
`README.md`). Attention (single-chip microarchitecture) and MoE routing
(multi-chip system architecture) are both done. This one — disaggregated
serving + weight/KV-cache placement — is the memory-hierarchy layer
underneath both: given a serving system split into prefill and decode
pools, what actually lives in SRAM vs. HBM vs. gets transferred, and how
does data move as prefill hands off to decode. Original spec:
[`spec.md`](spec.md). **Superseded/extended by
[`spec_v2.md`](spec_v2.md)**, which adds the MoE leg (Phase 2b) — read
`spec_v2.md`, not `spec.md`, for the current plan; `spec.md` stays as
historical record.

## Where things stand

**Phase 0 through Phase 4 are all done.** Phase 5 (full cross-project
synthesis, the capstone) is next, untouched. Read [`notes.md`](notes.md) in
full before doing anything — it's the actual record and this summary will
go stale the moment more work happens. Headline state, so you don't have to
read notes.md just to answer "what chip / what precision / what's the
ratio / what's the placement policy":

- **Simulator design** (Phase 0): discrete-event. Prefill machines, decode
  machines, one finite intermediate KV pool. Mooncake-style handoff (prefill
  machine frees immediately on ship, not DistServe's pull-from-GPU-memory
  model). N concurrently-active requests per decode machine, memory-bounded.
- **Chip: TPU 8i, homogeneous**, FP4 uniform, both pools — for the **dense**
  leg. A heterogeneous TPU 8i/Groq split was tried and deliberately reversed
  in Phase 0 (real reasoning kept in `notes.md`, not scrubbed).
- **Phase 1 (dense, Llama-3-70B)**: weights = 35GB; KV cache = 81,920
  bytes/token; N ≈ 320–339 concurrent max-length (8,192-token) requests/chip.
- **Phase 2a (dense chip ratio) — authoritative number**: **~5.82 prefill
  chips per decode chip** (`notes.md` §2.9) — corrected mid-Phase-2b after a
  real audit found dense's attention numbers were SDPA-only, missing QKVO
  projection FLOPs/bytes (21.4% of FFN's magnitude) throughout.
- **Phase 2b (MoE chip ratio, DeepSeek-V2 on TPU 8i)** — two real corrections
  after the first pass, read `notes.md` §2b.19–§2b.23 in order before
  trusting any single ratio number:
  - First-pass result (**~1.31 prefill chips per decode chip**, §2b.16),
    using DeepSeek-V2's real 163,840-token context cap, turned out to be
    **mostly a context-length artifact** — rerun at a cap matched to dense's
    8,192 (§2b.19), the ratio came back **~5.97:1, essentially identical to
    dense's ~5.82:1**. §2b.20 decomposed this further: swapping dense FFN →
    MoE's sparse FFN alone drops the ratio 5.82→4.60; swapping GQA → MLA
    alone pushes it back 4.60→5.97 — two real, *opposing* effects that
    nearly cancel, not one neutral effect.
  - **MLA is a context-length-conditional bet, not a universal
    optimization** (§2b.23): loses to plain GQA by ~9% at the matched 8,192
    cap, wins by **2.25×** at DeepSeek-V2's real 163,840 cap. Generalizes:
    any fixed architectural trade only pays off in the regime it was
    calibrated for.
  - **Deployment model: expert-parallel sharding**, 8-device EP group
    (§2b.4) — "one MoE decode machine" = one 8-chip EP group, not one
    atomic chip. Decode-N: 640/system (80/device) at the real 163,840 cap;
    12,864/system (1,608/device) at the matched 8,192 cap.
- **Phase 2c (KV handoff mechanism) — done, all numbers in `notes.md`
  §2c.1–§2c.11**:
  - Transfer cost: dense **40 MiB/request**; MoE (MLA) **8.4375 MiB/request**
    (4.74× smaller — MLA's KV-cache formula, `(d_c+d_h^R)×n_layers`, pulled
    fresh from the DeepSeek-V2 paper and independently confirmed against the
    figure already carried from Phase 2b).
  - Fabric: reused Boardfly (real, sourced TPU 8i interconnect) — the
    project has never claimed Google runs disaggregation on TPU 8i
    specifically, only that the fabric itself is real; that distinction is
    what let this be a non-issue.
  - KV handoff is **bandwidth-dominated** (17.48µs dense, 3.69µs MoE),
    unlike MoE dispatch traffic on the same fabric which is
    latency-dominated — a real regime difference from the payload being
    ~16,000× bigger.
  - Pool placement: locality-aware clustering (small repeated
    prefill:decode units, e.g. 2 boards/8 chips ≈ 6:1) beats a pod-wide
    half/half split, given a real unpublished hop-depth gap above the
    4-chip/board tier — resolved via a hop-count sweep (1–3 hops ×
    300–936.25ns), same move project #3 made for its own unpublished ICI
    latency.
  - DistServe sanity check: handoff cost is 0.29–0.34% (dense) and
    0.040–0.071% (MoE) of a single decode step — comfortably clears
    DistServe's own `<0.1%` claim (MoE outright; dense against a
    deliberately stricter single-step denominator than DistServe's own
    full-request one). Real mechanism: disaggregation helps via
    **interference avoidance** (DistServe's 60ms→200ms colocation finding),
    not because transfer is fast — cheap transfer just keeps separation
    from giving that win back.
- **Phase 3 (KV/weight placement policy) — done, all numbers in `notes.md`
  §3.1–§3.11**:
  - **Cap behavior: hard stop** (not compaction/sliding-window) — already
    the assumption baked into N=320 since Phase 1, and matches this
    project's chatbot-shaped workload (avg ~576 tokens, ~14× below the cap).
  - **Admission: hard-cap (N=320-family), not dynamic** — sensitivity
    computed anyway (N=4,558): throughput and chip ratio come back
    essentially unchanged (830.5 vs. 830.59 req/s/chip), because N=320
    already sits past the compute-bound crossover — a flat asymptote.
  - **Intermediate pool: kept block-until-space-frees**, with its real cost
    stated explicitly (re-couples prefill to decode's pace under sustained
    backpressure, partially undoing Mooncake's own decoupling rationale) —
    quantifying how often it binds needs Phase 4's real simulator.
  - **KV-cache quantization (KIVI-style, 2-bit) and CPU-offload: declined**
    — real, well-published, but quantization would ripple through every
    number since Phase 1 and reintroduces the requantization-boundary
    complexity already declined once via the Groq/FP4-FP8 reversal;
    CPU-offload declined for the same scope-expansion reason as the pool
    decision.
  - **Hot-expert SRAM residency — the open thread `spec_v2`'s Phase 2b
    section flagged, now closed**: under EP-sharding, each device's local
    shard is just 22 experts (259.5 MB) — fits under TPU 8i's 384 MB SRAM,
    but only *after* a naive attention-scratch estimate (56–112 MB,
    genuinely uncertain) was resolved by pulling FlashAttention's real
    tiling behavior directly (fused kernels reuse a small, fixed SRAM
    buffer sequentially, never materializing the whole batch's scores —
    the naive tight-fit concern was a modeling artifact, not real). **The
    whole local shard fits — not a hot/cold ranking problem after all.**
    Payoff: **zero at matched-cap** (already compute-bound), **~4.5×
    throughput at real-cap** (601.55 → 2,735 req/s/chip, N=80/device, was
    memory-bound 10.04×). Real-cap chip ratio recovers from **1.31:1 to
    ~5.97:1** — residency closes almost the entire gap to the matched-cap
    architecture comparison. The "2,735 ≈ 2,735.03" match between
    real-cap+residency and matched-cap+streaming is not a coincidence —
    both reach the identical compute-bound throughput ceiling via different
    routes (removing a fixed cost vs. brute-force batching past the
    crossover), the same asymptote mechanism the admission-policy
    sensitivity check hit independently.
- **Phase 4 (validate, discrete-event simulator) — done, all numbers in
  `notes.md` §4.1–§4.10**. Dense leg only (hard-cap admission, matching
  §3.3's actual chosen policy) — MoE and a real dynamic-admission/eviction
  model are legitimate follow-on passes, not built. Code in
  [`sim/`](sim/) (SimPy, from scratch — no prior Python convention existed
  in this repo). Both genuine unknowns `handoff.md` flagged for this phase
  are now resolved:
  - **Pool contention: essentially never happens at any realistically-
    provisioned size.** The "one round" sizing anchor (worst-case
    simultaneous full-batch completion across all 29 prefill machines,
    ≈38.92GB) showed *exactly zero* contention across 2M+ requests —
    confirmed real (not a wiring bug) via a deliberate tiny-pool (84MB)
    control test that *did* collapse the system. The actual threshold sits
    far below the original sizing logic: 500MB is unstable/bimodal, 1–2GB
    carries small-but-real contention (~1%) only once arrival rate exceeds
    the system's real ceiling, sub-1GB is where it becomes genuinely
    dangerous. Refined across two exploration passes at increasing
    statistical power — the picture sharpened, not reversed.
  - **Dynamic admission (N≈4,558) vs. hard-cap (N=320): zero measurable
    queueing benefit**, on top of §3.3's already-established zero
    throughput benefit — because neither cap is ever actually the thing a
    request waits on.
  - **The real, previously-unidentified finding: the system's actual
    bottleneck is prefill's fixed compute-bound throughput ceiling
    (~4,138 req/s across the 29-machine pool), not decode capacity and not
    the pool.** Decode occupancy plateaus under sustained overload
    (self-limiting — bounded by how fast prefill can feed it, confirmed via
    direct occupancy instrumentation) while prefill wait time grows
    unboundedly under the same load — the real congestion signature, found
    by building the simulator rather than assumed going in.
  - Building the sim caught three real bugs before any result was trusted:
    a wrong attention service-time formula in this project's own verbal
    design discussion (prefill attention is actually marginally
    compute-bound, not memory-bound — caught by a planning sub-agent,
    independently re-verified bit-exact against `notes.md`'s own numbers),
    a `simpy.Resource` race condition that could have silently over-admitted
    past the decode cap, and a missing pool-release call that would have
    made the pool monotonically fill and never drain.
  - External validation against DistServe/Mooncake is partial, stated
    honestly: the KV-handoff-latency claim was already checked analytically
    in Phase 2c and stands; the throughput-multiplier/SLO-attainment
    comparisons need a colocated-baseline simulator this pass explicitly
    didn't build (flagged, not overlooked); order-of-magnitude latency
    plausibility (the one thing checkable without a baseline) lands in a
    believable regime.

## How this project actually works — read before doing anything

This is the user's own learning project, not a delegate-and-execute task.
Match this operating mode or you're doing it wrong even if the technical
output is correct:

- **The user does the derivations, judgment calls, and design decisions.
  You do reference reading, real-source lookups, arithmetic *only when
  explicitly delegated*, and keeping `notes.md` current.** Don't pre-derive
  the interesting parts and hand over a finished answer. When the user is
  mid-derivation, ask the next sharpening question rather than completing
  the thought for them. **That said, the user sometimes explicitly wants to
  co-derive arithmetic in real time ("show me the derivation for my
  learning") — when they say that, walk through the math step by step with
  them rather than batching it into one big computed answer; when they
  don't say that, default delegation (compute it, show the result) is
  fine.** Calibrate to what they ask for in the moment, not a fixed rule
  either way.
- **Push back with a real, reasoned recommendation when asked "what do you
  think" — don't just list neutral options.** Reverse a position only when
  there's a real technical reason, and say so plainly either way. Phase 3
  did this twice: recommending hard-stop over compaction (workload-match
  reasoning, not just "simpler"), and recommending researching real
  fused-kernel behavior over immediately reaching for a hot/cold-expert
  ranking workaround (the naive SRAM estimate turned out to be the wrong
  model, not a real constraint).
- **The single most load-bearing discipline this project needs: a reused
  number or formula needs re-verification for the *new* context, not just
  confirmation that the *formula* is right.** Caught repeatedly across every
  phase: T=8,192, the SDPA-only attention scope, DeepSeek-V2's real context
  length (Phase 2b); and in Phase 3, the naive "weights fit in SRAM"
  capacity check that didn't account for what else SRAM has to hold
  *simultaneously* during compute — resolved only by checking real kernel
  behavior (FlashAttention), not by assuming either direction.
- **Ground everything in a real source.** This project pulled MLA's real
  KV-cache formula from the DeepSeek-V2 paper directly (Phase 2c), and
  FlashAttention's real tiling algorithm directly (Phase 3), in both cases
  *after* a secondhand or naive estimate had already produced a plausible
  but wrong-turning-out number. Reused numbers (chip specs, `seq_len=8192`,
  etc.) should come from this repo's own prior projects before reaching for
  a new source — but "came from a prior project" is necessary, not
  sufficient; still check it fits the new use case.
- **Catch errors by asking, not silently fixing** — this goes both ways:
  the user catches real errors by asking pointed questions (the "why do we
  even need those approaches if 22 experts obviously fit" pushback in Phase
  3 led directly to finding the naive-estimate/real-kernel gap), and
  self-caught errors get surfaced the same way, transparently, including
  walking back something said a few turns earlier when a formula or
  assumption doesn't hold up. Don't over-correct something the user already
  had right, either.
- **When the user explicitly delegates a low-stakes choice** ("just pick
  one, not something I want to spend time on"), make a reasoned pick, state
  the reasoning briefly, and log it — don't keep Socratic-questioning
  something they've said they don't want to spend time on.
- **Watch for scope creep against `spec.md`/`spec_v2.md`'s own stated
  philosophy** (focused projects, depth over breadth). Phase 3 hit this
  twice: declining real CPU/SSD tiered offload (a whole new memory tier to
  characterize) and declining KV-cache quantization below FP4 (would ripple
  through every downstream number) — both real, both flagged as legitimate
  future work, neither adopted, for stated reasons rather than by default.
- **Keep `notes.md` current as you go, not as a final wrap-up step** — log
  decisions (and reversals, kept on record rather than scrubbed) as they
  happen. The Phase 2c and Phase 3 conversations both batched the
  `notes.md` write-up at the end instead (explicit user choice that
  session, not the default), then split it into several small,
  logically-scoped commits rather than one large one — matches this
  project's "the more commits the merrier" preference when batching. Phase
  4 did the same — batched write-up at the end, explicit user choice again.
- **Two different collaboration modes coexist, and Phase 4 is where this
  became explicit** — the derivation-heavy phases (0–3) default to asking
  the next sharpening question rather than completing the user's thought.
  Phase 4's actual implementation work (writing the simulator) ran under a
  different, explicitly-granted mode instead: "you code end-to-end, only
  loop me in if there's actually something for me to learn or an
  interesting decision." Real design/methodology forks (batching semantics,
  pool-sizing derivation, what experiment to run next, whether a surprising
  result is a bug or a finding) still got surfaced and discussed either way
  — what changed was routine implementation (writing dispatcher code,
  tests, debugging a race condition) proceeding without asking first.
  Calibrate to which kind of work is actually in front of you, and don't
  assume the quieter mode extends to judgment calls just because it was
  granted for execution.

## Reading order for a new session

1. `spec_v2.md` — the current assignment (supersedes `spec.md`, which stays
   as historical record — Phase 0/1/2a-dense are unchanged by v2).
2. `notes.md` — full record of everything done so far; this is ground
   truth. If short on time, the "Key Findings" subsections closing out each
   phase (§2.7, §2b.18, §2c.11, §3.11) are the highest-signal summaries.
3. Top-level `README.md`, `prefill_notes.md`, `decode_notes.md`,
   `moe-routing-notes.md` — as needed, when a specific number or finding
   from the prior two projects gets referenced. Don't re-read all of these
   upfront; pull from them when `notes.md` points at them. This project has
   repeatedly found real gaps in trusting these sibling projects' numbers at
   face value for its own purposes — reread the specific section being
   reused, not just the headline number.
4. [`sim/`](sim/) (only if a specific Phase 4 number/mechanism needs
   re-checking against the actual code, not by default) — `disagg_sim/`
   for the entity model and formulas, `run_sweep.py`/`explore_*.py` for how
   each sweep was actually run, `results/*.csv` for raw data behind
   `notes.md` §4's numbers.

## Immediate next step

**Phase 5 — full cross-project synthesis, the capstone.** Per `spec_v2.md`
(supersedes `spec.md`'s three-legged version): the connective argument
across all **four** pieces of derived evidence now in hand — attention
(single-chip microarchitecture) → MoE routing (multi-chip interconnect) →
this project's dense disaggregation leg → this project's MoE disaggregation
leg. Given a fixed chip/power/area/bandwidth budget for a whole rack, how
should it split across (a) on-chip SRAM for attention scratchpad/
accumulator, (b) on-chip SRAM for hot MoE expert weights and KV cache, (c)
interconnect bandwidth for MoE dispatch, (d) interconnect bandwidth for
prefill→decode KV handoff. `spec_v2.md` flags one thing this synthesis owes
a paragraph to on its own, independent of the rack-budget question: the
dense-vs-MoE chip-ratio contrast from Phase 2 (§2.9's ~5.82:1 vs. ~5.97:1,
nearly identical via two opposing effects that cancel) is a direct,
quotable answer to "does disaggregation work the same way once you're
serving a MoE model" — the question that motivated `spec_v2.md`'s existence
in the first place.

Real material now available for this synthesis that wasn't when spec_v2 was
written: Phase 4 found the dense leg's actual bottleneck is prefill's
compute-bound ceiling, not decode capacity or the KV pool (§4.7/§4.10) —
worth checking whether that finding's shape (which resource is *actually*
scarce vs. generously provisioned) generalizes to how the rack-budget
question should be framed, rather than assuming SRAM/bandwidth are the only
axes that matter.

Start by confirming with the user how they want to scope the synthesis
(a written argument only, per spec_v2's own description, or something with
its own supporting derivation/validation) — don't just start writing, same
pattern every phase so far has opened with.
