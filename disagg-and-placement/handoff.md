# Handoff — read this first in a new conversation

Written because the conversation that produced Phase 0 and Phase 1 got long
enough to need a fresh chat. This file's only job is to get a new session
oriented fast and working the same way the old one did. It is not a project
doc — nothing here should get cited from `notes.md`; if something here turns
out to matter long-term, it belongs in `notes.md`, not here.

## What this is

Third of three sequential codesign projects in this repo (see top-level
`README.md`). Attention (single-chip microarchitecture) and MoE routing
(multi-chip system architecture) are both done. This one — disaggregated
serving + weight/KV-cache placement — is the memory-hierarchy layer
underneath both: given a serving system split into prefill and decode pools,
what actually lives in SRAM vs. HBM vs. gets transferred, and how does data
move as prefill hands off to decode. Full spec: [`spec.md`](spec.md).

## Where things stand

**Phase 0 (setup) and Phase 1 (memory hierarchy characterization) are done.**
Phase 2 (disaggregation hypothesis: chip ratio + handoff mechanism) is next,
untouched. Read [`notes.md`](notes.md) in full before doing anything —
it's the actual record and this summary will go stale the moment more work
happens. Headline state, so you don't have to read notes.md just to answer
"what chip / what precision":

- **Simulator design** (Phase 0): discrete-event, not closed-form. Entities:
  prefill machines, decode machines, one finite intermediate KV pool.
  Lifecycle: queue → prefill → ships to pool, prefill machine frees
  immediately (Mooncake-style, not DistServe's pull-from-GPU-memory model) →
  queue for decode capacity → joins as one of N concurrently-active requests
  on a decode machine, memory-bounded → steps to completion.
- **Chip: TPU 8i, homogeneous**, both pools. A heterogeneous TPU 8i/Groq
  split was tried and *deliberately reversed* — real reasoning, not a
  correction of a mistake, kept in full in `notes.md` under "Chip choice."
  Don't re-litigate this without reading why first.
- **Precision: uniform FP4** throughout (TPU 8i's native format).
- **Phase 1 numbers**: weights = 35GB (fit on one chip, no sharding);
  KV cache = 81,920 bytes/token (all 80 layers, GQA); aggregate KV-cache
  capacity per chip ≈ 2.63–2.78M tokens after a 10–15% fragmentation
  reserve (PagedAttention-sourced, not MoE's 46% — activations were ruled
  out as HBM-resident, see `notes.md` §1.3 for why); N ≈ 320–339 concurrent
  max-length (8,192-token) requests per decode chip.

## How this project actually works — read before doing anything

This is the user's own learning project, not a delegate-and-execute task.
Match this operating mode or you're doing it wrong even if the technical
output is correct:

- **The user does the derivations, judgment calls, and design decisions.
  You do reference reading, real-source lookups, arithmetic *only when
  explicitly delegated*, and keeping `notes.md` current.** Don't pre-derive
  the interesting parts and hand over a finished answer. When the user is
  mid-derivation, ask the next sharpening question rather than completing
  the thought for them.
- **Push back with a real, reasoned recommendation when asked "what do you
  think" — don't just list neutral options.** The user has explicitly
  checked, mid-project, whether a recommendation was just social
  accommodation ("you might have been pressured... if you think X is the
  play, be firm"). Answer that kind of check honestly; reverse a position
  only when there's a real technical reason, and say so plainly either way.
- **Ground everything in a real source** — a paper, an official spec page, a
  number already derived elsewhere in this repo. Flag sourcing quality
  explicitly (e.g. "official rack-level figure, per-chip number is exact
  division" vs. "secondary-sourced only, not yet primary-confirmed") rather
  than presenting everything as equally certain. Reused numbers (chip specs,
  `seq_len=8192`, etc.) should come from this repo's own prior projects
  before reaching for a new source, for cross-project comparability.
- **Catch errors by asking, not silently fixing.** Several real mistakes got
  caught this way (a dimensional mismatch in a capacity equation, an
  over-broad "worst case" fear that a hard cap actually already resolved) —
  in each case the fix was a pointed question, not a silent correction.
- **When the user explicitly delegates a low-stakes choice** ("just pick
  one, not something I want to spend time on"), make a reasoned pick, state
  the reasoning briefly, and log it — don't keep Socratic-questioning
  something they've said they don't want to spend time on.
- **Watch for scope creep against `spec.md`'s own stated philosophy**
  (three focused projects, depth over breadth, explicitly warns against one
  project trying to do another project's job). The Groq reversal happened
  specifically because pursuing it would have smuggled a chip-microarchitecture
  problem into a disaggregation/placement project.
- **Keep `notes.md` current as you go, not as a final wrap-up step** — log
  decisions (and reversals, kept on record rather than scrubbed) as they
  happen, the same Prediction/Log style `prefill_notes.md`/`decode_notes.md`
  describe using during their own live derivation.

## Reading order for a new session

1. `spec.md` — the actual assignment.
2. `notes.md` — full record of everything done so far; this is ground truth.
3. Top-level `README.md`, `prefill_notes.md`, `decode_notes.md`,
   `moe-routing-notes.md` — as needed, when a specific number or finding
   from the prior two projects gets referenced (KV-cache bytes formulas,
   roofline/ridge-point numbers, TPU 8i specs, etc.). Don't re-read all of
   these upfront; pull from them when `notes.md` points at them.

## Immediate next step

**Phase 2**: hypothesize a prefill:decode chip ratio, and the KV-cache
handoff mechanism (transfer cost, and which interconnect fabric — simpler
now than it would have been under the rejected Groq split, since both pools
share TPU 8i's Boardfly fabric already used in the MoE project). Start by
asking the user how they want to approach the chip-ratio question, the same
way Phase 0/1 started — don't just compute an answer.
