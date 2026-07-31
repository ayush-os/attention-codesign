# Handoff — Resume Here

**Purpose:** this doc exists so a new chat session can pick up this project exactly where the last one left off, without needing to re-read the full `notes.md` history first. Read this doc first; use `notes.md` as the detailed reference/derivation trail when you need it.

**Status as of this handoff:** Phase 1a complete, Phase 1b complete. Phase 1c was run end-to-end in a separate session that got destroyed (repo rename killed the Claude Code chat mid-flight) — but the Timeloop artifacts and Docker container survived, and were fully recovered and logged in `notes.md` under "Phase 1c: Timeloop Sweep — Recovered State." Two long-running mapper jobs (`_v3`, over 2 days of runtime) found during recovery were killed after confirming they'd already hit the theoretical utilization ceiling. **Next step: compare the recovered Phase 1c results against the Phase 1b hypothesis and explain the two open findings logged in notes.md** — that comparison has not been done yet.

---

## Critical collaboration constraint — read before responding to anything

This is a **self-directed learning project**. The user is going from workload characteristics to hardware design decisions, validated in RTL, and is doing every derivation themselves. Your job is to **check reasoning, ask questions that expose gaps, flag missing considerations, and help structure/log conclusions — not to solve steps, hand over derivations, or supply numbers/answers the user hasn't produced themselves.**

- When the user shares a hypothesis or derivation: probe assumptions, ask what they checked vs. asserted, don't supply the missing piece.
- Exception: pure arithmetic plug-in *after* the user has established the formula themselves — the user has explicitly said they're fine delegating that (they don't want to be a human calculator once the reasoning is done). Still don't derive the formula itself for them.
- Exception: factual/reference lookups (real hardware specs, tool defaults, "did we already log X") — verify via search rather than trusting memory, but don't extend this into doing their conceptual work.
- Phase 0/tooling setup is explicitly marked 🔧 in the project spec (boilerplate, not learning-bearing) — fine to help directly and concretely there, unlike the 🧠-marked conceptual phases.
- Continue logging into `notes.md` in the same Prediction/Log style rather than starting new files (except this handoff doc, which is meant to be overwritten/updated at each natural pause point, not appended to).
- The user works well with concrete, quantified counterexamples/questions (e.g., "would an 8192×8192 array be physically buildable?") rather than abstract pushback — ground Socratic questions in numbers where possible.

---

## Project structure (from `spec.md`)

Phase 0 (Timeloop/Accelergy + Chipyard/Gemmini setup) → **Phase 1 (prefill, compute-leaning)** → Phase 2 (decode, memory-leaning) → Phase 3 (numerics/precision) → optional Phase 4 (real-hardware check). Phase 1/2 each follow: 1a hand-derive FLOPs/bytes/AI/ridge-point → 1b hypothesize PE array/dataflow/scratchpad → 1c sweep Timeloop, compare → 1d configure Gemmini, read generated RTL, run via Verilator, explain every gap vs. Timeloop.

---

## Workload (locked in since Phase 1a)

Llama 3-70B prefill, from *How to Scale Your Model* (TPU serving chapter): batch=32, seq_len=8192, n_heads=64, n_kv_heads=8 (GQA), d_head=128, int8 throughout. Two passes done: naive MHA (control baseline) and GQA.

## Phase 1a results (full derivation in `notes.md`)

- Total FLOPs (MHA = GQA, proven identical): 2^46
- Bytes-moved: fused bound 8 GiB (MHA) / 4.5 GiB (GQA); unfused bound ≈520 GiB (MHA) / ≈516.5 GiB (GQA)
- AI: fused 8192 (MHA) / ≈14,564 (GQA); unfused ≈126 (MHA) / ≈126.9 (GQA)
- Ridge point: ≈480.5 FLOPs/byte (TPU v5e int8, same source as workload)
- Conclusion: fused → decisively compute-bound; unfused → decisively memory-bound. Regime is determined by the fusion (P-matrix on-chip) decision, not workload shape alone.
- GQA's benefit is regime-dependent in two distinct ways: (1) Amdahl's-Law style — its 8× K/V byte win barely shows up in the unfused case (P-traffic dominates) but shows up fully in fused; (2) roofline-position style — in fused/compute-bound prefill, GQA's bytes savings don't move execution time at all (time is set by FLOPs, unchanged by GQA); its real prefill payoff is scratchpad pressure and KV-cache footprint, not throughput. The throughput win is expected in memory-bound decode (Phase 2).
- Full "Key Takeaways" section logged in `notes.md` for the final writeup — see the Phase 1b takeaways section (17 points) for the fuller, more recent set, since several of Phase 1a's original 7 got extended/superseded by later findings.

## Phase 1b final hypothesis (full derivation and "Major open finding" section in `notes.md`)

- **PE array**: 128×128 primary (matches Trainium precedent), 128×256 carried as an explicit alternate to test.
- **Dataflow**: weight-stationary, K and V as the stationary operand (driven by the 8× group-reuse factor, `num_q_heads/num_k_heads`); Q streamed. One physical array serves both QK^T and ·V because `d_head` anchors the spatial pair in both (proven via first-principles spatial/temporal GEMM-dimension analysis, not assumed).
- **Scratchpad** (Gemmini real default, ≤1 MiB): double-buffered K/V chunks (`tile_k`=1024 — re-solved once, still 1024, now justified by `seq_len_k`=8192=2¹³ divisibility rather than just "power of 2 feels right") + fixed-size P tile (`tile_q×128`, fine-grained) + Q tile + output tile.
- **Accumulator** (Gemmini real default, ≤256 KiB): per-head online-softmax tracking state (running max + running sum + partial-output accumulator, fp32) for the group of 8 heads sharing one KV head — `tile_q`=32 — plus a transient raw S/P block (`tile_q×128`, ≈16 KB, one reused buffer not ×8).
- **Softmax granularity: fine-grained** (per 128-wide array sub-pass, not per full `tile_k` chunk) — switched from an initial coarse assumption after discovering coarse doesn't actually fit the accumulator budget (over by 2 KB once the raw-S term was counted honestly).
- Softmax's own ops (max/subtract/exp/sum/divide/rescale) don't touch the systolic array at all — need a separate vector/scalar unit; whether Gemmini has one natively or needs the host Rocket/BOOM core is an open Phase 1d question.

### Major open finding — read this carefully before starting Phase 1c

`tile_q`=32 was solved assuming only **one Q-tile's** worth of accumulator state at a time (no factor for holding multiple Q-tiles simultaneously). This forces **Q-tile to be the outer loop relative to K/V-chunk** in the full loop nest — you must finish one Q-tile (sweeping all 8 K/V-chunks) before starting the next. Consequence: each K/V-chunk gets **re-fetched from HBM ~256× (once per Q-tile)**, not once per group as Phase 1a's compulsory-bytes lower bound assumed — breaking the GQA "fetch once" reuse claim for *this specific* hand-derived config.

This is not considered a broken hypothesis — Phase 1a's own bytes-moved section explicitly predicted this category of gap in advance ("real mappings may re-read data if scratchpad can't hold what's needed... a predicted source of divergence from Timeloop/Gemmini"). It was deliberately **not hand-optimized further** (a smaller `tile_q` would trade instruction count for better reuse — a real, unexplored lever) and instead left as a sharpened, falsifiable prediction for Phase 1c: does Timeloop's mapper converge to something like `tile_q`=32 (accepting heavy re-fetching), or find a smaller `tile_q` that trades instructions for reuse?

### Second open finding — accumulator capacity as a free parameter (partially resolved)

While discussing the finding above, considered raising the accumulator past Gemmini's 256 KB *default* to relieve the tension (bigger accumulator → bigger `tile_q` or multiple simultaneous Q-tiles → less re-fetching). Initially reached for TPU v1's real, verified 4 MB accumulator as justification — caught that this is the wrong comparison basis (TPU v1 is a full custom datacenter ASIC with a completely different area budget than a small RTL generator; same category of mistake as the earlier TPU 8t/8i comparison, which was also rejected).

**Confirmed via a real Gemmini paper figure (user-provided):** a published, benchmarked "BigSP" SoC config exists with **512 KB scratchpad + 512 KB accumulator** (+1 MB L2), alongside a third real config ("BigL2," bigger L2 cache instead). So 512 KB is a real, realistic target for Phase 1c/1d, not just a plausible guess. But that same figure showed BigSP's measured speedup on the Matmul benchmark category (closest to attention) was small (~1-3%) versus Conv's (~10-11%) — a real reason not to assume "bigger accumulator" proportionally fixes the re-fetch tension. Still log accumulator capacity as an explicit free parameter for Timeloop's 1c sweep (128 KB base through the confirmed 512 KB BigSP point, possibly higher), alongside array shape and `tile_k`/`tile_q` — the exact optimum is still for Timeloop to find, but the search range is now grounded in a real config rather than speculative.

---

## Immediate next step: Phase 1c interpretation (tooling/runs are done)

Phase 1c tooling and runs are **complete** — see `notes.md`'s "Phase 1c: Timeloop Sweep — Recovered State" section for the full artifact inventory, results table, and raw winning mappings for QK^T and ·V. What's left is the actual 🧠 comparison-and-gap-explanation step the spec calls for, not further tool setup:

1. Compare the recovered Timeloop results against the Phase 1b hand hypothesis point by point — array shape, dataflow (was K/V actually found stationary?), scratchpad/accumulator sizing.
2. Resolve the two open findings logged in `notes.md` (flagged, not explained yet):
   - QK^T's winning map keeps nothing resident in scratchpad at all (no K/V-stationary reuse found), yet still hits the ideal 100%-utilization compute-bound cycle count despite the modeled DRAM traffic including the full unfused 128 GiB P-write — is Phase 1a's TPU-v5e-derived ridge point (≈480.5) even the right comparison for an architecture that only models a single 128×128 array, or does this architecture have its own, different ridge point?
   - ·V's winning dataflow differed qualitatively between the two completed runs (output-stationary @ 100% util vs. weight-stationary @ 80.63% util) even though only the clock/bandwidth modeling was changed between them, not the architecture — mapper search variance, or a real effect of the bandwidth fix?
3. If a complete `_v3` (larger search budget) record is wanted for the writeup, rerun bounded (e.g. `search_size=375000`) rather than open-ended — the previous `_v3` attempt ran for 2+ days and was killed after confirming it wasn't going to find a different answer (see notes.md for why).

Per the project's own framing, explaining these mismatches mechanistically is the highest-value part of Phase 1c/1d — don't just note that they diverge.

---

## Where everything else lives

- `spec.md` — original project spec, all phases.
- `notes.md` — full derivation trail for Phase 1a, 1b, and the recovered Phase 1c results, in Prediction/Log style. This is the artifact to keep appending to.
- `timeloop-accelergy-exercises/workspace/attention_1c/` (separate repo/docker environment, not this one) — the actual Timeloop configs and run outputs referenced above.
- This file (`handoff.md`) — update/overwrite at the next natural pause point rather than leaving stale.
