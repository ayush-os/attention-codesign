# Handoff — read this first in a new conversation

Written because the conversation that produced Phase 2b got long enough to
need a fresh chat. This file's only job is to get a new session oriented
fast and working the same way the old one did. It is not a project doc —
nothing here should get cited from `notes.md`; if something here turns out
to matter long-term, it belongs in `notes.md`, not here.

## What this is

Third of three sequential codesign projects in this repo (see top-level
`README.md`). Attention (single-chip microarchitecture) and MoE routing
(multi-chip system architecture) are both done. This one — disaggregated
serving + weight/KV-cache placement — is the memory-hierarchy layer
underneath both: given a serving system split into prefill and decode
pools, what actually lives in SRAM vs. HBM vs. gets transferred, and how
does data move as prefill hands off to decode. Original spec:
[`spec.md`](spec.md). **Superseded/extended by
[`spec_v2.md`](spec_v2.md)**, which adds the MoE leg (Phase 2b) this
session just finished — read `spec_v2.md`, not `spec.md`, for the current
plan; `spec.md` stays as historical record.

## Where things stand

**Phase 0, Phase 1, Phase 2a, and Phase 2b are all done.** Phase 2c (KV
handoff mechanism) is next, untouched. Read [`notes.md`](notes.md) in full
before doing anything — it's the actual record and this summary will go
stale the moment more work happens. Headline state, so you don't have to
read notes.md just to answer "what chip / what precision / what's the
ratio":

- **Simulator design** (Phase 0): discrete-event. Prefill machines, decode
  machines, one finite intermediate KV pool. Mooncake-style handoff (prefill
  machine frees immediately on ship, not DistServe's pull-from-GPU-memory
  model). N concurrently-active requests per decode machine, memory-bounded.
- **Chip: TPU 8i, homogeneous**, FP4 uniform, both pools — for the **dense**
  leg. A heterogeneous TPU 8i/Groq split was tried and deliberately reversed
  in Phase 0 (real reasoning kept in `notes.md`, not scrubbed).
- **Phase 1 (dense, Llama-3-70B)**: weights = 35GB; KV cache = 81,920
  bytes/token; N ≈ 320–339 concurrent max-length (8,192-token) requests/chip.
- **Phase 2a (dense chip ratio) — corrected mid-Phase-2b, this is the
  authoritative number**: **~5.82 prefill chips per decode chip**
  (`notes.md` §2.9), not the earlier ~5.50 — a real audit found dense's
  attention numbers (reused from `prefill_notes.md`/`decode_notes.md`) were
  explicitly SDPA-only, missing QKVO projection FLOPs/bytes (21.4% of FFN's
  magnitude) throughout. Fixed, logged, mechanistically explained (an
  Amdahl's-Law-shaped reason the ratio moved only a little despite a real
  correction — FFN's dominant share of each phase's own service time
  diluted the fix differently per phase).
- **Phase 2b (MoE chip ratio, DeepSeek-V2 on TPU 8i) — done, headline
  result**: **~1.31 prefill chips per decode chip** (`notes.md` §2b.16) —
  dramatically more balanced than dense's ~5.82:1, and for a real,
  mechanistic reason, not just a different coefficient: dense's high ratio
  is almost entirely a decode-batching windfall (crossing FFN into
  compute-bound at N=320) that MoE can't reach at its own real HBM capacity
  (its crossover needs N≈807/device; capacity only supports N≈80/device),
  while MoE's prefill is independently cheaper per chip from sparse
  routing's lower compute/token. Both effects push the ratio the same
  direction. Full derivation chain in `notes.md` §2b.1–§2b.18 (§2b.18 is the
  Key Findings summary — read that first if short on time).
  - **Deployment model: expert-parallel sharding** (8-device EP group per
    `moe-routing-notes.md`'s own real deployment), reversed from an initial
    full-replication choice after the replication numbers turned out
    structurally misleading (§2b.2–§2b.4). "One MoE decode machine" =
    one 8-chip EP group, not one atomic chip (dense stays single-chip).
  - **Decode-N regrounded to 640** (not project #3's borrowed 8,192, not
    the dense-derived 320) — DeepSeek-V2's own real shipped context length
    (163,840, verified via HF config, not the borrowed 8,192) run through
    this project's own 10–15% reserve convention (§2b.7).
  - **Prefill uses naive MLA (explicit K/V), decode uses absorbed** — a
    real architectural fork resolved by mechanism (absorption only pays off
    across *repeated* re-materialization; prefill pays that cost once,
    §2b.12), not a "same formula, different N" substitution.

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
  co-derive arithmetic in real time ("I wanna be in the loop on this one")
  — when they say that, walk through the math step by step with them rather
  than batching it into one big computed answer; when they don't say that,
  default delegation (compute it, show the result) is fine.** Calibrate to
  what they ask for in the moment, not a fixed rule either way.
- **Push back with a real, reasoned recommendation when asked "what do you
  think" — don't just list neutral options.** Reverse a position only when
  there's a real technical reason, and say so plainly either way. This
  session reversed the deployment-model recommendation (replication →
  sharding) after finding two real reasons (full replication guts Phase 3's
  hot-expert-residency question; the Groq-style "scope creep" fear that
  drove the original replication pick didn't actually transfer, since
  project #3 had already done the hard sharding-topology work) — a genuine
  reversal, not deference to the user's preference.
- **The single most load-bearing discipline this project needs: a reused
  number or formula needs re-verification for the *new* context, not just
  confirmation that the *formula* is right.** This session caught three
  separate real instances purely by asking "does this actually apply here":
  T=8,192 (a real project's real number, still wrong for this project's
  capacity purpose), the SDPA-only attention scope (completely correct for
  the sibling attention project's own question, silently incomplete once
  reused for throughput modeling), and DeepSeek-V2's real context length
  (neither the borrowed 8,192 nor an assumed round number — checked via HF
  config). None were formula errors — all were *scope/context* mismatches,
  the harder kind to catch. Keep asking this question by default, even
  about numbers that already came from a rigorous prior project.
- **Ground everything in a real source** — a paper, an official spec page
  (this session pulled DeepSeek-V2's real HF config rather than trust a
  recalled number), a number already derived elsewhere in this repo. Flag
  sourcing quality explicitly rather than presenting everything as equally
  certain. Reused numbers (chip specs, `seq_len=8192`, etc.) should come
  from this repo's own prior projects before reaching for a new source —
  but "came from a prior project" is necessary, not sufficient; still check
  it fits the new use case (see above).
- **Catch errors by asking, not silently fixing** — and this goes both
  ways: the user caught real errors this session by asking pointed
  questions (the N/T mismatch, the SDPA-only gap), and self-caught errors
  should be surfaced the same way, transparently, including when it means
  walking back something said a few turns earlier (e.g. reconsidering the
  causal-masking treatment once the actual precedent was checked). Don't
  over-correct something the user already had right, either — one instance
  this session got called out plainly ("that's what I said") after
  dressing up an already-correct point in more formal language; own it
  and move on rather than re-litigating.
- **When the user explicitly delegates a low-stakes choice** ("just pick
  one, not something I want to spend time on"), make a reasoned pick, state
  the reasoning briefly, and log it — don't keep Socratic-questioning
  something they've said they don't want to spend time on. (Used this
  session for the Zipf-vs-two-tier distribution choice, the prefill batch
  size, and the causal-discount question.)
- **Watch for scope creep against `spec.md`/`spec_v2.md`'s own stated
  philosophy** (focused projects, depth over breadth). The Groq reversal in
  Phase 0, and the deployment-model reasoning in Phase 2b, both hinge on
  this same test: does pursuing X actually smuggle a different project's
  job into this one, or does it just look that way at first glance?
- **Keep `notes.md` current as you go, not as a final wrap-up step** — log
  decisions (and reversals, kept on record rather than scrubbed) as they
  happen. This session logged ~18 subsections during live derivation
  (§2b.1–§2b.18), plus a real-time correction to dense's own "done" Phase 2a
  numbers (§2.9) once the QKVO gap was found mid-Phase-2b.

## Reading order for a new session

1. `spec_v2.md` — the current assignment (supersedes `spec.md`, which stays
   as historical record — Phase 0/1/2a-dense are unchanged by v2).
2. `notes.md` — full record of everything done so far; this is ground
   truth. If short on time, `notes.md` §2.7 (Phase 2a Key Findings) and
   §2b.18 (Phase 2b Key Findings) are the two highest-signal summaries.
3. Top-level `README.md`, `prefill_notes.md`, `decode_notes.md`,
   `moe-routing-notes.md` — as needed, when a specific number or finding
   from the prior two projects gets referenced. Don't re-read all of these
   upfront; pull from them when `notes.md` points at them. Note: this
   session found real gaps in trusting these sibling projects' numbers at
   face value for *this* project's purposes (see "How this project works,"
   above) — reread the specific section being reused, not just the
   headline number.

## Immediate next step

**Phase 2c**: the KV-handoff mechanism (transfer cost, interconnect
fabric) — spec_v2's Phase 2c, carried over unchanged from spec v1, **still
not started**. Two things already flagged in `notes.md` §2b.17 to carry in,
not rediscover:

- **MoE's KV cache is 4.74× smaller per token than dense's** (17,280 vs.
  81,920 bytes/token, MLA compression vs. GQA) — needs its own transfer-cost
  number, not a reused dense one.
- **MoE's handoff needs to target one specific device within the 8-chip EP
  group** (attention is data-parallel, not replicated, so only one device
  can own a given request's KV cache) — a decision surface dense's
  single-chip handoff never had. The mechanism itself is standard
  sticky-session load balancing (shared queue, pull on a free slot among
  ~80 concurrent-request capacity/device) — not a hard problem, just new to
  MoE. Don't over-invent a solution here; the answer is expected to be
  unsurprising.

Start by asking the user how they want to approach the interconnect-fabric
question (same Boardfly fabric MoE project #3 already validated as
latency-dominated, or a dedicated channel — spec's own open question) —
don't just compute an answer, same pattern Phase 0/1/2a/2b all started
with.
