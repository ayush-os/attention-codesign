# Handoff — Resume Here

**Purpose:** this doc exists so a new chat session can pick up this project exactly where the last one left off, without needing to re-read the full `phase1_notes.md` history first. Read this doc first; use `phase1_notes.md` as the detailed reference/derivation trail when you need it.

**Status as of this handoff:** Phase 1a complete, Phase 1b complete (with one major open finding). Phase 1c not started — that's the next step.

---

## Critical collaboration constraint — read before responding to anything

This is a **self-directed learning project**. The user is going from workload characteristics to hardware design decisions, validated in RTL, and is doing every derivation themselves. Your job is to **check reasoning, ask questions that expose gaps, flag missing considerations, and help structure/log conclusions — not to solve steps, hand over derivations, or supply numbers/answers the user hasn't produced themselves.**

- When the user shares a hypothesis or derivation: probe assumptions, ask what they checked vs. asserted, don't supply the missing piece.
- Exception: pure arithmetic plug-in *after* the user has established the formula themselves — the user has explicitly said they're fine delegating that (they don't want to be a human calculator once the reasoning is done). Still don't derive the formula itself for them.
- Exception: factual/reference lookups (real hardware specs, tool defaults, "did we already log X") — verify via search rather than trusting memory, but don't extend this into doing their conceptual work.
- Phase 0/tooling setup is explicitly marked 🔧 in the project spec (boilerplate, not learning-bearing) — fine to help directly and concretely there, unlike the 🧠-marked conceptual phases.
- Continue logging into `phase1_notes.md` in the same Prediction/Log style rather than starting new files (except this handoff doc, which is meant to be overwritten/updated at each natural pause point, not appended to).
- The user works well with concrete, quantified counterexamples/questions (e.g., "would an 8192×8192 array be physically buildable?") rather than abstract pushback — ground Socratic questions in numbers where possible.

---

## Project structure (from `codesign_project_spec.md`)

Phase 0 (Timeloop/Accelergy + Chipyard/Gemmini setup) → **Phase 1 (prefill, compute-leaning)** → Phase 2 (decode, memory-leaning) → Phase 3 (numerics/precision) → optional Phase 4 (real-hardware check). Phase 1/2 each follow: 1a hand-derive FLOPs/bytes/AI/ridge-point → 1b hypothesize PE array/dataflow/scratchpad → 1c sweep Timeloop, compare → 1d configure Gemmini, read generated RTL, run via Verilator, explain every gap vs. Timeloop.

---

## Workload (locked in since Phase 1a)

Llama 3-70B prefill, from *How to Scale Your Model* (TPU serving chapter): batch=32, seq_len=8192, n_heads=64, n_kv_heads=8 (GQA), d_head=128, int8 throughout. Two passes done: naive MHA (control baseline) and GQA.

## Phase 1a results (full derivation in `phase1_notes.md`)

- Total FLOPs (MHA = GQA, proven identical): 2^46
- Bytes-moved: fused bound 8 GiB (MHA) / 4.5 GiB (GQA); unfused bound ≈520 GiB (MHA) / ≈516.5 GiB (GQA)
- AI: fused 8192 (MHA) / ≈14,564 (GQA); unfused ≈126 (MHA) / ≈126.9 (GQA)
- Ridge point: ≈480.5 FLOPs/byte (TPU v5e int8, same source as workload)
- Conclusion: fused → decisively compute-bound; unfused → decisively memory-bound. Regime is determined by the fusion (P-matrix on-chip) decision, not workload shape alone.
- GQA's benefit is regime-dependent in two distinct ways: (1) Amdahl's-Law style — its 8× K/V byte win barely shows up in the unfused case (P-traffic dominates) but shows up fully in fused; (2) roofline-position style — in fused/compute-bound prefill, GQA's bytes savings don't move execution time at all (time is set by FLOPs, unchanged by GQA); its real prefill payoff is scratchpad pressure and KV-cache footprint, not throughput. The throughput win is expected in memory-bound decode (Phase 2).
- Full "Key Takeaways" section (7 points) logged in `phase1_notes.md` for the final writeup.

## Phase 1b final hypothesis (full derivation and "Major open finding" section in `phase1_notes.md`)

- **PE array**: 128×128 primary (matches Trainium precedent), 128×256 carried as an explicit alternate to test.
- **Dataflow**: weight-stationary, K and V as the stationary operand (driven by the 8× group-reuse factor, `num_q_heads/num_k_heads`); Q streamed. One physical array serves both QK^T and ·V because `d_head` anchors the spatial pair in both (proven via first-principles spatial/temporal GEMM-dimension analysis, not assumed).
- **Scratchpad** (Gemmini real default, ≤1 MiB): double-buffered K/V chunks (`tile_k`=1024 — re-solved once, still 1024, now justified by `seq_len_k`=8192=2¹³ divisibility rather than just "power of 2 feels right") + fixed-size P tile (`tile_q×128`, fine-grained) + Q tile + output tile.
- **Accumulator** (Gemmini real default, ≤256 KiB): per-head online-softmax tracking state (running max + running sum + partial-output accumulator, fp32) for the group of 8 heads sharing one KV head — `tile_q`=32 — plus a transient raw S/P block (`tile_q×128`, ≈16 KB, one reused buffer not ×8).
- **Softmax granularity: fine-grained** (per 128-wide array sub-pass, not per full `tile_k` chunk) — switched from an initial coarse assumption after discovering coarse doesn't actually fit the accumulator budget (over by 2 KB once the raw-S term was counted honestly).
- Softmax's own ops (max/subtract/exp/sum/divide/rescale) don't touch the systolic array at all — need a separate vector/scalar unit; whether Gemmini has one natively or needs the host Rocket/BOOM core is an open Phase 1d question.

### Major open finding — read this carefully before starting Phase 1c

`tile_q`=32 was solved assuming only **one Q-tile's** worth of accumulator state at a time (no factor for holding multiple Q-tiles simultaneously). This forces **Q-tile to be the outer loop relative to K/V-chunk** in the full loop nest — you must finish one Q-tile (sweeping all 8 K/V-chunks) before starting the next. Consequence: each K/V-chunk gets **re-fetched from HBM ~256× (once per Q-tile)**, not once per group as Phase 1a's compulsory-bytes lower bound assumed — breaking the GQA "fetch once" reuse claim for *this specific* hand-derived config.

This is not considered a broken hypothesis — Phase 1a's own bytes-moved section explicitly predicted this category of gap in advance ("real mappings may re-read data if scratchpad can't hold what's needed... a predicted source of divergence from Timeloop/Gemmini"). It was deliberately **not hand-optimized further** (a smaller `tile_q` would trade instruction count for better reuse — a real, unexplored lever) and instead left as a sharpened, falsifiable prediction for Phase 1c: does Timeloop's mapper converge to something like `tile_q`=32 (accepting heavy re-fetching), or find a smaller `tile_q` that trades instructions for reuse?

---

## Immediate next step: Phase 1c tooling setup

Phase 0 tooling (Timeloop/Accelergy via Docker, Gemmini/Chipyard via Verilator) is assumed done — confirm with the user if picking this up fresh. For Phase 1c specifically, four artifacts are needed (this part is 🔧, fine to help build directly):

1. **Workload/problem spec** (Timeloop YAML) — separate specs for QK^T (M=`seq_len_q`, N=`seq_len_k`, K=`d_head`) and ·V (M=`seq_len_q`, N=`d_head`, K=`seq_len_k`), with batch/`num_q_heads`/`num_k_heads` as outer loop bounds.
2. **Architecture spec** — PE array as a *parameterized* search space (so Timeloop can sweep 128×128 vs. 128×256 and others, not locked to the hand hypothesis), scratchpad and accumulator as separate memory levels with parameterized capacity (sweep around the ~1 MiB / 256 KiB hand estimates), HBM level with TPU v5e bandwidth/capacity from the Phase 1a ridge-point work.
3. **Accelergy energy/area models** for each component (PE, scratchpad, accumulator, DRAM interface).
4. **Mapper/constraint config** — dataflow permutations unconstrained (see if the mapper *independently* finds K/V-stationary), tiling factor ranges for `tile_q`/`tile_k`, legal loop orderings.

Goal for 1c: find Timeloop's near-optimal config and compare against the Phase 1b hypothesis above — where do they agree, where do they diverge, and (per the spec) explaining the mismatch is the important part, not just noting it.

---

## Where everything else lives

- `codesign_project_spec.md` — original project spec, all phases.
- `phase1_notes.md` — full derivation trail for Phase 1a and 1b, in Prediction/Log style. This is the artifact to keep appending to.
- This file (`handoff.md`) — update/overwrite at the next natural pause point (e.g., end of Phase 1c) rather than leaving stale.
