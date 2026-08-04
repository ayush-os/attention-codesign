# Handoff — Resume Here

**Purpose:** this doc exists so a new chat session can pick up this project exactly where the last one left off, without needing to re-read the full derivation trail first. Read this doc first; use `prefill_notes.md` and `decode_notes.md` as the detailed reference when needed.

**Status as of this handoff:** **Phase 1 (prefill) and Phase 2 (decode, through 2b + cross-phase comparison) are both complete.** Phase 2c/2d/4 were deliberately skipped (real, documented reasons — not gaps); Phase 3 was reframed rather than run as originally specified, and is the natural next step if this project continues. Full scope reasoning: `decode_notes.md` §4, `spec.md`'s "Amendment" section.

---

## Critical collaboration constraint — read before responding to anything

This is a **self-directed learning project**. The user is going from workload characteristics to hardware design decisions, doing every derivation themselves. Your job is to **check reasoning, ask questions that expose gaps, flag missing considerations, and help structure/log conclusions — not to solve steps, hand over derivations, or supply numbers/answers the user hasn't produced themselves.**

- When the user shares a hypothesis or derivation: probe assumptions, ask what they checked vs. asserted, don't supply the missing piece. Ground Socratic questions in concrete numbers where possible (this worked well throughout Phase 2 — e.g. quantifying the systolic-array fill/drain problem against the actual roofline margin, not just "8 cycles sounds small").
- Exception: pure arithmetic plug-in *after* the user has established the formula themselves. Also fine: presenting a derived formula/setup for the user to plug numbers into, once the underlying structure has been jointly established (used throughout Phase 2b's sizing derivations).
- Exception: factual/reference lookups (real hardware specs, tool defaults, Gemmini/Chisel source behavior, real vector/SIMD engine widths, "did we already log X") — verify via web search/source reading rather than trusting memory. Phase 2b used this to confirm Gemmini has no SIMD/vector compute path at all (`decode_notes.md` §2.6) and to pull real precedent for SIMD lane widths (AVX-512, TPU VPU, GPU warp) — both direct, load-bearing uses of this exception.
- Phase 0/tooling setup and organizational/logging work (writing up `decode_notes.md`, editing `spec.md`'s amendment, refactoring notes into a polished writeup) are explicitly fine to do directly — that's the "help structure/log conclusions" part of the job, not the 🧠 conceptual part.
- **Check your own proposed structure against precedent before asserting it, and be willing to correct yourself when the user catches a gap.** This came up twice in Phase 2b: the user correctly caught that "SRAM chunk size" (C) actually depends on "lane count" (D) — the reverse of what was initially proposed — by tracing through how Phase 1b's own P-terms were actually sized. Don't just accept a proposed ordering/structure at face value; re-derive it from precedent when asked, and own it clearly when the correction is right.
- **Scope is a legitimate, discussable topic, not just something to push through.** Phase 2 had two real scope moments: (a) mid-derivation, checking whether Gemmini could even represent a SIMD-based hardware hypothesis *before* refining it further (it can't — real, load-bearing finding, `decode_notes.md` §2.6), and (b) after 2b, an explicit discussion of whether 2c/2d/3/4 were worth running at all, resolved with direct recommendations and reasoning (not just reflected back as an open question) — `decode_notes.md` §4. When a phase's marginal learning value looks low relative to its time/setup cost, or a tool can't actually represent the hypothesis being tested, surface that directly with a real recommendation, the way both of those were handled.
- **Don't assume a prior phase's constraint transfers "for consistency" without checking whether its original justification still applies.** The strip-mining finding (`decode_notes.md` §2.5) is the clean example: Phase 1b's power-of-2 tiling constraint existed to avoid needing ragged-chunk-handling hardware — but decode already needs that hardware anyway (for the ramp-up case), so the constraint didn't need to carry over. Checking *why* a prior decision was made, not just *what* it was, is what surfaced this.
- **Logging convention**: for any future phase (e.g. Phase 3 numerics), keep a live working log in a fresh `notes.md` (Prediction/Log style, prediction stated before checking, log entry once resolved) the same way Phase 1 and Phase 2 were both tracked, then refactor into a polished `<phase>_notes.md` at that phase's natural completion point — mirroring how both `prefill_notes.md` and `decode_notes.md` were produced. Update this `handoff.md` file (overwrite, don't append) at each natural pause point.

---

## Project structure (from `spec.md`, plus its "Amendment" section for actual executed scope)

Phase 0 (setup) → Phase 1 (prefill, full 1a→1d loop — **complete**) → Phase 2 (decode: 2a workload characterization + 2b hardware hypothesis + cross-phase comparison — **complete**; 2c/2d **deliberately skipped**, reasoning in `decode_notes.md` §2.6/§4) → Phase 3 (numerics — **reframed, not yet executed**, see below) → Phase 4 (optional real-hardware check — **skipped**, reasoning in `decode_notes.md` §4).

---

## Phase 1 (prefill): complete — summary

**Full derivation in `prefill_notes.md`.** Do not re-derive any of this — it's a finished, validated record. The compressed version: Llama 3-70B prefill (batch=32, seq_len=8192, n_heads=64, n_kv_heads=8, d_head=128, int8). Regime (compute- vs. memory-bound) is governed by whether the softmax/P intermediate is fused on-chip — fused is decisively compute-bound (AI 8192/14564 ≫ ridge 480.5), unfused is decisively memory-bound (AI ≈126 ≪ ridge). Hardware hypothesis: 128×128 systolic array, K/V-stationary weight-stationary dataflow (GQA-group-reuse-driven), `tile_k=1024`, `tile_q=32` — with a major finding that `tile_q=32`'s accumulator capacity forces sequential per-head processing, breaking full GQA reuse (~256× re-fetch instead of once per group). Validated in Timeloop (1c, confirmed ridge point is architecture-specific, not a fixed workload property) and Gemmini/Verilator RTL (1d, confirmed native softmax hardware matches independently-derived online-softmax, confirmed axis-routing via the transposer, found "WS-only" restricts behavior via a control constant rather than generating smaller hardware).

---

## Phase 2 (decode): complete through 2b + cross-phase comparison — summary

**Full derivation in `decode_notes.md`.** Do not re-derive any of this. The compressed version:

- **Workload**: same config as prefill, but `seq_len` splits into `seq_len_q=1` (new token) and `seq_len_kv=8192` (context length) — a structural change, not a substitution.
- **2a**: FLOPs collapse 8,192× vs. prefill (driven entirely by `seq_len_q`). Bytes only drop ~2×(MHA)/~9×(GQA) vs. prefill, because K/V bytes are literally unchanged between regimes (depend only on `seq_len_kv`) — the entire byte gap is Q/output, which collapse to near-zero. AI: MHA≈2, GQA≈16, both decisively below the 480.5 ridge (~240×/~30× margins) — far more decisive than prefill's memory-bound case ever was. GQA flips from secondary (prefill) to first-order (decode's only lever within SDPA, since fusion isn't meaningful here — P is too small to ever be the bottleneck).
- **2b**: systolic array rejected (fill/drain amortization fails at `seq_len_q=1` — a naive utilization estimate lands right at the edge of decode's roofline margin, checked quantitatively, not assumed). Hypothesis: SIMD/vector engine, lanes tiled across `seq_len_kv` (not `d_head` — the independent axis, not the reduction axis), 32 lanes (real precedent 32–128+, logged as low-sensitivity). Full 8-head GQA group reuse confirmed achievable (~63× accumulator margin, unlike prefill's forced compromise) — accumulator-before-lane-count-before-SRAM-chunk was a real, corrected ordering dependency, traced through Phase 1b's own precedent. `tile_kv_sram=2043` via strip-mining (reusing ramp-up's masking hardware to avoid prefill's power-of-2 markdown to 1024) — a real, quantified divergence from Phase 1b's approach, not just "different numbers."
- **Gemmini tool-representability gap** (§2.6): confirmed via source that Gemmini has no SIMD/vector compute path at all — only a systolic array. This structurally blocks Phase 2d (worse than Phase 1c's fusion-modeling gap, since here the *entire* compute primitive isn't representable, not just one piece).
- **Cross-phase comparison** (§3, the spec's explicit Phase 2 deliverable ask): every hardware difference between prefill's 1b and decode's 2b traces to the single root cause of `seq_len_q` collapsing from 8192→1 — not independent choices. Matches the AI numbers precisely: prefill's hardware chases throughput (17× above ridge), decode's hardware avoids wasting silicon on unusable compute capability (30–240× below ridge).
- **Scope** (§4): 2c/2d skipped (tool gap + low marginal ROI given Phase 1c/1d's real cost). Phase 4 skipped (already substantively tested twice — Phase 1d's real RTL work, and prior *How to Scale Your Model* work). Phase 3 reframed.

---

## Immediate next step: Phase 3 (numerics), reframed

Per `decode_notes.md` §4: **not the spec's literal version** (precision-mode throughput comparison on Gemmini — would be generic and hits the same Gemmini-representability wall as 2d). The recommended, not-yet-executed version: a hand-derivation-only question, no tool validation needed (same style as 2a/2b) —

**Is KV-cache quantization (below int8) a bigger lever on decode's AI than GQA was?**

Two real threads motivate this rather than it being a forced addition:
1. Phase 1a explicitly flagged precision as a "carried to Phase 3" open thread (`prefill_notes.md` §1.2 — P's int8 requantization-per-round-trip tradeoff, never resolved).
2. This repo's sibling MoE project already found, for a different severely memory-bound workload, that **numerics — not imbalance — was the dominant lever that actually moved the regime** (top-level `README.md`: "floor scales inversely with dispatch precision, crossover ≈2.5 bytes/element"). Decode attention is now a second severely memory-bound, bytes-dominated workload — worth checking whether the same pattern holds. Connects directly to real Rivos FP8/MXFP8 production background, not a generic exercise.

If picked up: start the same way every phase has — hand-derive a prediction (does going below int8 for K/V move AI proportionally? is there a crossover point analogous to the MoE project's ≈2.5 bytes/element?) before checking anything, log it in a fresh `notes.md`, Prediction/Log style, refactor into a polished doc at completion.

---

## Where everything lives

- `spec.md` — original project spec, all phases, plus an "Amendment" section documenting actual executed scope for Phase 2 (2c/2d/4 skipped, 3 reframed) with reasoning.
- `prefill_notes.md` — complete, polished Phase 1 record. Reference only.
- `decode_notes.md` — complete, polished Phase 2 record (2a, 2b, cross-phase comparison, scope reasoning, open threads). Reference only — nothing here should be re-derived.
- `notes.md` — does not currently exist; **create fresh** if Phase 3 (or anything else) is picked up, same Prediction/Log style used for both prior phases.
- This file (`handoff.md`) — update/overwrite at the next natural pause point rather than leaving stale.
