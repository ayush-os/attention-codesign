# Handoff — Resume Here

**Purpose:** this doc exists so a new chat session can pick up this project exactly where the last one left off, without needing to re-read the full `notes.md` history first. Read this doc first; use `notes.md` as the detailed reference/derivation trail when you need it.

**Status as of this handoff:** Phase 1a, 1b, and 1c are all complete. **Next step: Phase 1d** — configure Gemmini, read the generated RTL, validate via Verilator.

---

## Critical collaboration constraint — read before responding to anything

This is a **self-directed learning project**. The user is going from workload characteristics to hardware design decisions, validated in RTL, and is doing every derivation themselves. Your job is to **check reasoning, ask questions that expose gaps, flag missing considerations, and help structure/log conclusions — not to solve steps, hand over derivations, or supply numbers/answers the user hasn't produced themselves.**

- When the user shares a hypothesis or derivation: probe assumptions, ask what they checked vs. asserted, don't supply the missing piece.
- Exception: pure arithmetic plug-in *after* the user has established the formula themselves — the user has explicitly said they're fine delegating that (they don't want to be a human calculator once the reasoning is done). Still don't derive the formula itself for them.
- Exception: factual/reference lookups (real hardware specs, tool defaults, "did we already log X") — verify via search rather than trusting memory, but don't extend this into doing their conceptual work.
- Phase 0/tooling setup is explicitly marked 🔧 in the project spec (boilerplate, not learning-bearing) — fine to help directly and concretely there, unlike the 🧠-marked conceptual phases.
- Continue logging into `notes.md` in the same Prediction/Log style rather than starting new files (except this handoff doc, which is meant to be overwritten/updated at each natural pause point, not appended to).
- The user works well with concrete, quantified counterexamples/questions (e.g., "would an 8192×8192 array be physically buildable?") rather than abstract pushback — ground Socratic questions in numbers where possible.
- When something surfaces mid-investigation that's cheaper to check by reading existing files/logs than by reasoning it out or running something new (e.g. "did an old raw search log already show X?"), do that lookup directly rather than asking the user to re-derive it — this has repeatedly turned out to be faster and more conclusive than new runs or hand-waving. Doesn't extend to the actual conceptual interpretation of what's found.

---

## Project structure (from `spec.md`)

Phase 0 (Timeloop/Accelergy + Chipyard/Gemmini setup) → **Phase 1 (prefill, compute-leaning)** → Phase 2 (decode, memory-leaning) → Phase 3 (numerics/precision) → optional Phase 4 (real-hardware check). Phase 1/2 each follow: 1a hand-derive FLOPs/bytes/AI/ridge-point → 1b hypothesize PE array/dataflow/scratchpad → 1c sweep Timeloop, compare → 1d configure Gemmini, read generated RTL, run via Verilator, explain every gap vs. Timeloop.

---

## Workload (locked in since Phase 1a)

Llama 3-70B prefill, from *How to Scale Your Model* (TPU serving chapter): batch=32, seq_len=8192, n_heads=64, n_kv_heads=8 (GQA), d_head=128, int8 throughout.

## Phase 1a results (full derivation in `notes.md`)

- Total FLOPs (MHA = GQA, proven identical): 2^46
- Bytes-moved: fused bound 8 GiB (MHA) / 4.5 GiB (GQA); unfused bound ≈520 GiB (MHA) / ≈516.5 GiB (GQA)
- AI: fused 8192 (MHA) / ≈14,564 (GQA); unfused ≈126 (MHA) / ≈126.9 (GQA)
- Ridge point: ≈480.5 FLOPs/byte (TPU v5e int8, same source as workload) — **note this is the workload-source chip's ridge point, not necessarily the ridge point of whatever architecture is actually being modeled in a later phase; see Phase 1c finding #1 below, it does not automatically transfer.**
- Conclusion: fused → decisively compute-bound; unfused → decisively memory-bound. Regime is determined by the fusion (P-matrix on-chip) decision, not workload shape alone.
- GQA's benefit is regime-dependent in two distinct ways: (1) Amdahl's-Law style — its 8× K/V byte win barely shows up in the unfused case (P-traffic dominates) but shows up fully in fused; (2) roofline-position style — in fused/compute-bound prefill, GQA's bytes savings don't move execution time at all (time is set by FLOPs, unchanged by GQA); its real prefill payoff is scratchpad pressure and KV-cache footprint, not throughput. The throughput win is expected in memory-bound decode (Phase 2).
- Full "Key Takeaways" section logged in `notes.md` for the final writeup (Phase 1a: 7 points, Phase 1b: 17 points, Phase 1c: 4 points).

## Phase 1b final hypothesis (full derivation in `notes.md`)

- **PE array**: 128×128 primary (matches Trainium precedent), 128×256 carried as an explicit alternate.
- **Dataflow**: weight-stationary, K and V as the stationary operand (driven by the 8× group-reuse factor, `num_q_heads/num_k_heads`); Q streamed. One physical array serves both QK^T and ·V because `d_head` anchors the spatial pair in both.
- **Scratchpad** (Gemmini real default, ≤1 MiB): double-buffered K/V chunks (`tile_k`=1024) + fixed-size P tile (`tile_q×128`, fine-grained) + Q tile + output tile.
- **Accumulator** (Gemmini real default, ≤256 KiB): per-head online-softmax tracking state (running max + running sum + partial-output accumulator, fp32) for the group of 8 heads sharing one KV head — `tile_q`=32 — plus a transient raw S/P block (`tile_q×128`, ≈16 KB).
- **Softmax granularity: fine-grained** (per 128-wide array sub-pass, not per full `tile_k` chunk). Softmax's own ops don't touch the systolic array — need a separate vector/scalar unit; whether Gemmini has one natively or needs the host Rocket/BOOM core is an **open Phase 1d question**.
- **Major open finding carried through 1c**: `tile_q`=32 forces Q-tile outer to K/V-chunk in the loop nest, breaking the "fetch K/V once per group" GQA reuse assumption (each chunk re-fetched ~256× instead of once) — a real, hand-predicted-in-advance tension, deliberately left for Timeloop/Gemmini to resolve rather than hand-optimized further.
- Accumulator capacity was logged as a free sweep parameter (128 KB–512 KB, the 512 KB point grounded in a real published Gemmini "BigSP" config), not fixed at the 256 KB default.

---

## Phase 1c: complete — summary (full trail in `notes.md`)

Ran end-to-end in a since-destroyed session (repo rename killed the chat mid-flight); Timeloop artifacts and the Docker container survived and were fully recovered, then the comparison-and-gap-explanation step was completed against Phase 1b's hypothesis. Artifacts live in `timeloop-accelergy-exercises/workspace/attention_1c/` (**separate repo/docker environment, not this one** — problem specs for QK^T and ·V as conv-style GEMMs, an architecture translating the Phase 1b hypothesis into a sweepable Timeloop model, dataflow left unconstrained at the scratchpad so the mapper had to discover K/V-stationary on its own).

**Four findings, all resolved mechanistically — this is the part a fresh Phase 1d session actually needs:**

1. **Ridge point is a property of the specific accelerator being modeled, not a fixed workload characteristic.** QK^T hit 100% utilization even with the full *unfused* 128 GiB P-write traffic modeled — looks like it contradicts Phase 1a's "unfused should be memory-bound" prediction (AI≈126 vs. ridge 480.5), but this Timeloop architecture models a single 128×128 array at a 1 GHz base clock (not TPU v5e's 4 MXUs at 1.5 GHz) — 6× lower peak compute at the same real HBM bandwidth, so *this* architecture's own ridge point is ≈80.08, not 480.5. AI≈126 clears 80.08 — compute-bound was correct all along for this specific modeled system. **Generalizes**: any Phase 1a/1b roofline conclusion checked against a Phase 1c/1d tool result needs its ridge point recomputed for the actual modeled hardware, not reused from the workload's source chip.
2. **How much dataflow choice matters is itself regime-dependent.** ·V's winning dataflow differed between two runs that only changed the clock/bandwidth model (v1: 1 GHz; v2: 2 GHz "int8-pumped," DRAM bandwidth held fixed in absolute bytes/s — i.e. v2's peak compute is 2× v1's, meaning v2's *own* ridge point is 2× v1's). Confirmed via the identical mapping appearing in both raw search logs: under v1 it hit 100% utilization, under v2 (same mapping, unchanged) only 80.63% — a real DMA/compute-overlap effect, not noise. Near/below an architecture's own ridge (v1), competing dataflows were essentially fungible (weight- and output-stationary tied at 100% within a rounding error); once the ridge doubled (v2), the same mapping fell behind and which dataflow "wins" started to matter far more.
3. **A single mapper run's "winner" is not necessarily the true optimum — it's whatever a deterministic, budget-limited search finds.** Confirmed directly: rerunning `primary_v_v2`'s exact config produced a byte-for-byte identical search log and the same 80.63% answer — `random_pruned` is deterministic given a fixed config, not randomly seeded. The true 100% ceiling does exist under v2's architecture (seen in a separate, deeper, since-killed search's raw log), but escaping the 80.63% local optimum needs materially more search depth, not another attempt at the same depth — a methodology lesson for interpreting *any* Timeloop mapper result going forward, not just this one.
4. **This Phase 1c setup cannot model "fused" at all, structurally — not a mapper/architecture finding.** Every run's DRAM traffic shows the full P round-trip (137,438,953,472 B) regardless of dataflow, because QK^T and ·V are two independent Timeloop problems with no on-chip path connecting one's output to the next's input — no mapper search, however deep, can change that. This matters because **Phase 1b's entire scratchpad/accumulator sizing exercise was designed specifically to enable fusion** — meaning Phase 1c has only ever characterized the *unfused* regime. **Decision made: not worth hacking Timeloop to fake fusion** (its problem format isn't built for modeling two matmuls with an intermediate tensor that never leaves the chip) — **real fusion validation is deferred to Phase 1d**, where a Gemmini RTL pipeline can literally keep P in scratchpad between the two matmuls, no modeling hack required. An open, not-yet-computed footnote exists for a cheap hand-projected "fused-equivalent" estimate if wanted before Phase 1d lands (subtract shared P-traffic bytes/energy from the existing unfused numbers using Phase 1a's own byte formulas).

---

## Immediate next step: Phase 1d (Gemmini config + RTL validation)

Per the spec:
- Configure Gemmini as close as the generator allows to the Phase 1b/1c-validated config (128×128 PE array, weight-stationary K/V, 1 MiB scratchpad / 256 KiB accumulator).
- Actually read the generated RTL — trace how the array-size/dataflow choices show up as real datapath structure.
- Run the attention kernel (or a representative slice) through Verilator, get real cycle counts/utilization.
- Compare against the Phase 1c Timeloop numbers, and explain every meaningful gap mechanistically — same "gap-hunting is the highest-value activity" framing that paid off four times over in Phase 1c. Findings #3 and #4 above are natural first places to check for a Timeloop-vs-RTL gap (does real Gemmini's mapper/compiler hit the same local optimum? does the real hardware pipeline actually achieve fusion, and if so what does it cost relative to Phase 1c's unfused numbers?).
- Also still open from Phase 1b: whether Gemmini has a native vector/scalar unit for softmax, or whether it routes through the host Rocket/BOOM core (a real potential dispatch/data-movement overhead Timeloop's cost model wouldn't have captured).

---

## Where everything else lives

- `spec.md` — original project spec, all phases.
- `notes.md` — full derivation trail for Phase 1a, 1b, and 1c, in Prediction/Log style. This is the artifact to keep appending to.
- `timeloop-accelergy-exercises/workspace/attention_1c/` (separate repo/docker environment, not this one) — the actual Timeloop configs and run outputs referenced above.
- This file (`handoff.md`) — update/overwrite at the next natural pause point rather than leaving stale.
