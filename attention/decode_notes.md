# Decode Attention — Workload → Silicon (Phase 2 + Phase 3, Complete)

This is the consolidated, polished record of decode-phase attention (memory-leaning regime): Phase 2's hand-derived roofline analysis → hardware hypothesis → explicit cross-phase comparison against Phase 1's prefill hypothesis, plus Phase 3's reframed numerics question. Phase 2c (Timeloop) and 2d (Gemmini RTL) were deliberately scoped out, and Phase 4 (optional real-hardware check) skipped — all as reasoned, documented decisions rather than defaults or unfinished work (`spec.md`, "Amendment" section has the full scope reasoning). This document supersedes the working Prediction/Log-style derivations it was tracked as while the work was in progress — organized here as a finished writeup, mirroring how `prefill_notes.md` was itself refactored from a live log; self-corrections and real dependency-ordering mistakes are kept where they carried signal, pure process narration is cut.

---

## 0. Workload and Methodology

**Workload**: same Llama 3-70B GQA configuration as Phase 1, chosen for continuity (not re-derived from scratch, a stated choice):

| Parameter | Value | Note |
|---|---|---|
| batch | 32 | inert for SDPA's own AI (§1.2) — kept for continuity, not derived |
| seq_len_q | 1 | new: prefill's single `seq_len` splits into two independent params for decode |
| seq_len_kv | 8192 | "running length of the convo" — an already-filled context, same magnitude as prefill's `seq_len` |
| n_heads (query) | 64 | unchanged |
| n_kv_heads (GQA) | 8 | unchanged |
| d_head | 128 | unchanged |
| precision | int8 | unchanged |

**Scope**: this project's decode analysis covers SDPA only (QK^T → softmax → ·V), not the surrounding QKVO/FFN projection weights — matters directly for the batch discussion in §1.2 below.

**Matrix dimensions** (naive SDPA, per batch element & head — holds for both prefill and decode, only `seq_len_q`/`seq_len_kv` differ):

| Matrix | Shape |
|---|---|
| Q | `(seq_len_q, d_head)` |
| K | `(seq_len_kv, d_head)` |
| V | `(seq_len_kv, d_head)` |
| S = QKᵀ | `(seq_len_q, seq_len_kv)` |
| P = softmax(S) | `(seq_len_q, seq_len_kv)` |
| O = PV | `(seq_len_q, d_head)` |

At decode's `seq_len_q = 1`: Q, S/P, and O all collapse to vectors.

---

## 1. Phase 2a — Hand-Derived Roofline Analysis

### 1.1 FLOPs

QK^T FLOPs (output `(seq_len_q, seq_len_kv)`, contraction `d_head`) = `2 × seq_len_q × seq_len_kv × d_head`. PV FLOPs (output `(seq_len_q, d_head)`, contraction `seq_len_kv`) = `2 × seq_len_q × d_head × seq_len_kv`. **These are the same product written in different order — a general structural identity of SDPA's two matmuls (multiplication is commutative), true for any `seq_len_q`/`seq_len_kv`** — correcting Phase 1a's framing (`prefill_notes.md` §1.1), which attributed the match to "seq_len appearing symmetrically" specifically because prefill happened to have `seq_len_q = seq_len_kv`. Decode, where the two lengths differ, confirms the equality is general, not coincidental.

Per (batch, head): `2 × 1 × 8192 × 128 = 2²¹` FLOPs each for QK^T and PV. Scaled by batch × n_heads (`2¹¹`): **total FLOPs = 2³³ ≈ 8.59×10⁹ — identical for MHA and GQA** (GQA only changes bytes, never FLOPs, confirmed unchanged from Phase 1a's finding — FLOPs count query-head executions, not how many distinct KV tensors back them, and that logic transfers directly regardless of regime).

### 1.2 Bytes Moved

**Batch is structurally inert for SDPA's own AI.** Every batch element carries a distinct KV cache — no cross-batch reuse — so FLOPs and bytes scale together, linearly, with batch. Real decode-serving systems batch aggressively regardless, but that payoff lives in the FFN/projection weights (shared across batch), out of this project's SDPA-only scope.

**Fusion — prefill's dominant lever (65× AI swing) — is not a meaningful lever in decode.** With `seq_len_q = 1`, P/S collapses to `(1, 8192)` ≈ 8 KiB, trivially SRAM-resident regardless of fuse/unfuse choice. Decode has only one regime worth deriving, unlike prefill's fused/unfused pair.

**Per (batch, head), MHA** (no cross-head K/V sharing): load = `Q(128 B) + K+V(2×8192×128 B) = 2,097,280 B`; write (O) = `128 B`. Scaled by batch × n_heads (2048): **MHA total = 4,295,491,584 B ≈ 4.0005 GiB.**

**GQA** (K/V loaded once per KV-head group — the same "compulsory bytes, perfect on-chip reuse" idealization Phase 1a used for prefill): per batch, load = `n_heads×128 (Q) + n_kv_heads×8192×128×2 (K+V)`; write = `n_heads×128 (O)`. Scaled by batch (32): **GQA total = 537,395,200 B ≈ 0.5005 GiB.**

**MHA/GQA ratio ≈ 7.99×** — essentially the *full* theoretical 8× group-size reduction, vs. prefill's Amdahl's-Law-capped **1.78×** (`prefill_notes.md` §1.2). K and V bytes are *literally identical* between prefill and decode — both depend only on `seq_len_kv` (unchanged), never `seq_len_q`. The entire prefill-vs-decode byte gap, and the entire reason GQA's win is capped in one regime but not the other, is 100% attributable to Q and output — the only two tensors scaling with `seq_len_q` (8192 in prefill → 1 in decode, collapsing to near-zero, removing the term that capped prefill's GQA win).

### 1.3 Arithmetic Intensity and Ridge Point

| | AI (FLOPs/byte) |
|---|---|
| MHA | 2³³ / 4,295,491,584 ≈ **2.0** |
| GQA | 2³³ / 537,395,200 ≈ **15.98** |

**Ridge point**: 480.5 (TPU v5e, int8 — same reference chip as Phase 1a, for internal consistency). **Both decisively memory-bound** — MHA ~240× below ridge, GQA-improved ~30× below ridge. Far more decisive than prefill's unfused case (~3.8× below ridge).

**Resolves Phase 1a's pre-registered open thread directly** (`prefill_notes.md` Key Takeaway #7 / §6): fused prefill got zero throughput benefit from GQA (FLOPs-bound, GQA doesn't touch FLOPs). Decode shows the real payoff — an 8× AI jump (2 → 16) that reflects directly in real bytes-moved, because there's no Amdahl's-Law-capping term here to blunt it.

### 1.4 Cross-Phase Comparison: Prefill vs. Decode (FLOPs & Bytes)

| | Prefill | Decode | Ratio |
|---|---|---|---|
| FLOPs (MHA & GQA) | 2⁴⁶ ≈ 7.037×10¹³ | 2³³ ≈ 8.59×10⁹ | prefill is **8,192×** more |
| Bytes, MHA (prefill fused) | 8 GiB | ≈4.0005 GiB | prefill is **~2×** more |
| Bytes, GQA (prefill fused) | 4.5 GiB | ≈0.5005 GiB | prefill is **~9×** more |

(Decode compared against prefill's *fused* numbers — the fair comparison, since decode has no meaningful unfused case, §1.2.)

**The mechanism, precisely — not just "roughly from `seq_len_q`"**: K and V bytes are identical between regimes (both depend only on `seq_len_kv`); the entire byte gap is Q+O, the only tensors scaling with `seq_len_q`. Checked exactly: prefill MHA fused = Q(2)+K(2)+V(2)+O(2) GiB = 8 GiB; decode MHA ≈ K(2)+V(2)+negligible ≈ 4 GiB — the 2× gap *is* Q+O. Same for GQA: 4.5 GiB = Q(2)+K(0.25)+V(0.25)+O(2); decode's 0.5 GiB ≈ K(0.25)+V(0.25) alone.

**Core intuition, tying FLOPs and bytes together**: `seq_len_q` is the knob deciding how many FLOPs get to amortize each byte of K/V fetched — literally what arithmetic intensity measures. Prefill cranks it to 8192, decode collapses it to 1: same K/V-fetch bill in both cases, but prefill spreads it over 8,192× more compute. This single parameter is the entire mechanism behind landing 17× above the ridge point in one regime and 30–240× below it in the other.

### 1.5 Key Findings — Phase 2a

1. Decode splits prefill's single `seq_len` into two independent params (`seq_len_q=1`, `seq_len_kv`=context length) — a structural change to the workload's shape, not a substitution.
2. QK^T = PV FLOPs is a general structural identity (commutative product), not a shape-specific coincidence as originally framed in Phase 1a — a correction, not an overturn.
3. Batch is inert for SDPA's own AI by construction (no cross-batch KV reuse) — real decode batching's payoff lives in out-of-scope weight amortization.
4. Fusion, prefill's dominant lever, isn't meaningful in decode — P/S is too small to ever be the bottleneck either way.
5. GQA's byte-savings pass through almost fully in decode (~7.99× of 8×) vs. prefill's capped 1.78× — Q/O collapsing to negligible removes the capping term.
6. GQA flips from secondary (Phase 1a Key Takeaway #4: mattered only once fusion was solved) to first-order — the *only* lever within SDPA that moves decode's AI at all, with no fusion lever available. (Sparse/linear attention would be additional levers beyond GQA — flagged, out of scope.)
7. Decode is far more decisively memory-bound than prefill's memory-bound case ever was (~240×/~30× below ridge vs. ~3.8×) — opposite regimes, by very different margins.
8. `seq_len_q` is the single root parameter behind essentially everything found in 2a — the FLOPs collapse, the byte-gap mechanism — and, as §3 shows, the entire hardware-hypothesis divergence from prefill in §2.

---

## 2. Phase 2b — Hardware Hypothesis

### 2.1 Why Not a Systolic Array

A systolic array amortizes a one-time pipeline fill/drain cost (~`array_rows + array_cols` cycles, ~256 for 128×128) over a deep temporal stream of data pushed through one stationary load. In prefill, one stationary K/V chunk supported `group_size × seq_len_q = 8 × 8192 = 65,536` cycles of useful streaming before a reload — the fill/drain cost was invisible. In decode, the same stationary chunk supports only `group_size × seq_len_q = 8 × 1 = 8` cycles.

Checked quantitatively against the roofline margin itself, not just "8 sounds small": compute time at 100% utilization is `2³³/3.94×10¹⁴ ≈ 21.8 μs`; the memory-bound floor is `537,395,200/8.2×10¹¹ ≈ 655.4 μs` — a **~30× margin**. A naive fill/drain-dominated utilization estimate (~8 useful cycles per ~256-cycle load, ~3%, i.e. a ~30× slowdown from peak) lands **right at that edge** — the fill/drain inefficiency would consume essentially the entire roofline margin, leaving no slack for other real overhead (DMA latency, instruction issue effects — both found to matter in Phase 1c/1d). Betting on the systolic array being "fine anyway" was judged the riskier choice, given this project's own track record of gap-hunting revealing real overhead exactly at close calls like this one.

### 2.2 Compute Primitive and Parallelization Axis

**SIMD/vector engine, not a systolic array** — a SIMD lane's pipeline latency is a short, fixed constant, independent of stream depth, sidestepping the fill/drain problem structurally rather than trying to minimize it.

**Lanes tile across `seq_len_kv`, not `d_head`.** Each of the `seq_len_kv` score outputs (QK^T) is independent of every other — no cross-lane reduction needed — unlike `d_head`, the contraction/reduction dimension inside each individual score's dot product (the reason `d_head` was the *spatial* axis for the systolic array, and is the wrong axis to parallelize a SIMD engine across). `seq_len_kv = 8192 = 2¹³` is power-of-2-friendly, the same shape-divisibility property Phase 1b leaned on for `tile_k`.

**Ramp-up underutilization** (real, but self-resolving): actual context length starts small and grows monotonically over a conversation, so a fixed lane width is underutilized only for the first `tile_size_kv` decode steps of any conversation, never recurring after that. Given decode's ~30× roofline margin even at 100% utilization, a *smaller* lane width likely costs nothing in steady state (compute isn't the bottleneck) while shrinking this ramp-up window — an argument for erring small, not large, unlike prefill's throughput-driven array-shape reasoning.

**Per-lane reduction structure — deliberately left unresolved, not chased further**: each lane must still reduce over `d_head=128` for its own dot product. A dedicated adder tree (`log2(128)=7` cycles) was considered against naive serial per-lane accumulation (128 cycles), but given decode's large roofline margin, this level of microarchitectural detail is unlikely to change the headline conclusion — flagged as an open parameter rather than resolved by hand, the same move Phase 1b made with accumulator capacity and the 128×256 array-width alternate.

### 2.3 Loop Order, Reuse Goal, and Accumulator Sizing

**Goal**: sweep the GQA group's 8 heads against one resident K/V chunk before advancing to the next chunk (preserving GQA reuse) — the decode analog of prefill's "K/V-stationary across the group." **Explicitly not assumed obvious**, even though it's the "of course" answer: Phase 1b's own history (§2.5) shows this exact starting assumption broke for prefill under a real capacity constraint (forced `tile_q=32`, ~256× re-fetch, sequential per-head processing instead of concurrent). Stating this as a goal to check, not a settled fact, is what made the next step meaningful rather than redundant.

**Accumulator check**: per-head running-softmax state at decode's `seq_len_q=1` (vs. prefill's `tile_q`): `8 heads × 1 × (2 + 128) × 4 bytes = 4,160 B` — **~63× under the 256 KiB budget**. Unlike prefill, where this exact style of check forced a compromise, decode's version resolves cleanly: full 8-head concurrent GQA reuse is genuinely achievable, and because capacity supports it, all 8 heads' Q vectors and output accumulators are loaded resident simultaneously (not one head at a time, unlike prefill's forced sequential pattern) — a direct, necessary consequence of this check, not an independent design preference. Same structural risk pattern as Phase 1b, opposite resolution — purely because `seq_len_q` collapsed from `tile_q`-scale (up to 32) to 1. Resolves Phase 1's carried-over open thread ("whether decode's GQA byte-savings are actually achievable under a real scratchpad budget," `prefill_notes.md` §6) with a clean yes.

### 2.4 Lane Count

Reframed from prefill's "how big can we make it" (throughput-driven — prefill needed the parallelism to hit its roofline potential) to decode's "how small can this go while still comfortably clearing the memory-bound floor" — decode has ~30–240× roofline slack even at 100% utilization, so unlike prefill, bigger buys no throughput benefit, while smaller shrinks the ramp-up-underutilization window (§2.2) for free.

Real precedent checked (no Gemmini equivalent exists to cite — Gemmini has no vector engine at all, §2.6): AVX-512 = 64 int8 lanes, TPU VPU = `(8, 128)` shape, GPU warp = 32 threads — a wide range (32–128+), not a single forced answer the way Trainium anchored prefill's array size. **Logged as low-sensitivity**: because SIMD has no fill/drain penalty at any width, and decode's roofline margin is so large, no choice in this range changes any headline conclusion — it only feeds forward into sizing the S/P term in §2.5, itself a small fraction of the total budget in Phase 1b's analogous case. **Decided: 32** (smallest real precedent, GPU-warp-anchored, `8192/32=256` clean chunks), picked and stated rather than rigorously derived — consistent with how Phase 1b treated its own less-constrained choices (§2.6, Key Takeaway #6: "state a primary hypothesis... rather than forcing one answer").

### 2.5 SRAM-Resident Chunk Sizing — and a Real Ordering Correction

Initially sequenced the remaining work as loop-order → accumulator → SRAM-chunk-size → lane-count, mirroring Phase 1b's accumulator-before-scratchpad order at face value. **Corrected mid-derivation**: re-examining Phase 1b's actual `tile_k` budget (§2.3) showed both its P-related terms (`tile_q × 128` raw-S in the accumulator, `tile_q × 128` quantized-P in the scratchpad) were sized using the *array's spatial sub-pass width* (128) — decode's lane-count analog, not the scratchpad-chunk-size analog. So lane count had to be resolved *before* the SRAM budget could be written down, not after. Correct order: loop-order → accumulator → lane-count → SRAM-chunk-size.

**Budget** (four terms, direct analogy to Phase 1b's `tile_k` derivation): double-buffered K/V chunk (scales with `tile_kv_sram`, coefficient `512` — same as prefill's, since `d_head` is unchanged: `2` double-buffer `× 2` K&V `× 128 × 1 byte`) + Q for the 8-head group (`8×128×1 = 1,024 B`, fixed — ×8 because §2.3 already committed to holding all 8 heads concurrently, not an independent or optional choice) + output for the group (1,024 B, same shape as Q) + quantized-P term sized by lane count (`8×32×1 = 256 B`, fixed, not scaling with `tile_kv_sram` — mirrors Phase 1b's fine-grained-P becoming a fixed term once granularity was resolved).

`512 × tile_kv_sram + 2,304 ≤ 1,048,576` → **raw ceiling `tile_kv_sram ≤ 2,043.5`.**

**Real divergence from Phase 1b's precedent**: Phase 1b constrained its analogous raw ceiling (~2,024) down to a power-of-2 (`tile_k=1024`) specifically to guarantee `seq_len_k=8192`'s tiling divides evenly with no ragged final chunk. Decode doesn't need that constraint: the ramp-up case (§2.2) already requires masking/partial-chunk hardware regardless — early conversation turns have `seq_len_kv` smaller than a full tile, and that same mechanism handles a ragged tail at the end of a full-length sweep for free, since both are the identical underlying problem ("this chunk has fewer real elements than the tile's capacity") just occurring at different times. This is exactly **SIMD strip-mining** — processing full-width chunks plus a masked/predicated remainder pass, a real, standard hardware/compiler technique, not an invented workaround. **`tile_kv_sram = 2,043` stands directly**, without the power-of-2 markdown — nearly doubling effective SRAM utilization versus what blindly re-applying Phase 1b's own constraint would have forced (1,024).

### 2.6 Gemmini Tool-Representability Gap and Scope Decision

Verified directly against Gemmini's source (`ucb-bar/gemmini`, full `src/main/scala/gemmini/` file listing + README, mirroring Phase 1d's direct-source-verification style): **Gemmini has no general-purpose SIMD/vector compute path.** The systolic array (`Mesh`/`PE`/`Tile`) is its only matmul-capable structure; `dataflow` only selects OS-vs-WS *within* that same array (consistent with Phase 1d's §4.6 finding). `VectorScalarMultiplier.scala` looked like a candidate by name but is only a per-element scale/quantize pipeline used during DMA move-in, not a GEMM engine. Gemmini's own README states directly: *"At the heart of the accelerator lies a systolic array which performs matrix multiplications."*

**Consequence**: the Phase 2b hardware hypothesis (SIMD-based) is not representable by the Phase 2d validation toolchain at all — a harder gap than Phase 1c's fusion-modeling limitation, where the *rest* of the hypothesis besides fusion was still testable. Here the core compute-primitive choice itself isn't buildable in Gemmini.

**Scope decision, made explicitly rather than defaulted into**: given (a) this structural tool-representability gap and (b) direct prior experience that Phase 1c/1d cost 3–4 days of wall-clock time for comparatively little marginal learning versus 1a/1b's one day, **Phase 2d (Gemmini/RTL) was skipped entirely**, and **Phase 2c (Timeloop) was skipped as well** — its real payoff in Phase 1 was tool-methodology lessons (architecture-specific ridge points, mapper local-optima, tool-representability limits) already learned once and actively applied to Phase 2 without re-running the tool, against a decode workload whose conclusions are already more decisive (240×/30× margins) and more thoroughly hand-explained than prefill's were. Mirrors the project's own stated fallback framing (`spec.md`, "Fallback / Minimum Viable Version": Phase 1 alone was already declared a complete, defensible artifact) — extended here to Phase 2's 2a/2b loop, with the stopping point stated and justified rather than left implicit.

### 2.7 Final Hypothesis Summary

- **Compute primitive**: SIMD/vector engine, not systolic — fill/drain amortization fails structurally at `seq_len_q=1`.
- **Parallelization axis**: lanes tile across `seq_len_kv` (independent per-lane work, unlike `d_head`'s reduction role).
- **Lane count**: 32 — real precedent range 32–128+, logged as low-sensitivity, picked rather than rigorously derived.
- **Loop order / reuse**: full 8-head GQA group swept against one resident K/V chunk before advancing — confirmed achievable (~63× accumulator margin), unlike prefill's forced compromise.
- **SRAM-resident K/V chunk**: `tile_kv_sram = 2,043`, using strip-mining to avoid prefill's power-of-2 markdown.
- **Per-lane reduction structure**: deliberately left open (adder tree vs. serial), flagged as unlikely to matter.
- **Validation**: Phase 2c/2d both explicitly skipped, with reasoning (§2.6).

### 2.8 Key Findings — Phase 2b

1. The fill/drain problem is the single reason the hardware hypothesis diverges from prefill's at all — everything downstream (parallelization axis, lane count, chunk sizing) follows from having first rejected the systolic array as the compute primitive.
2. A quantitative check (comparing the naive utilization estimate against the roofline margin itself, not just "the number looks small") is what actually decided the systolic-vs-SIMD question — a qualitative "8 sounds small" wouldn't have been rigorous enough to commit to a real architectural pivot.
3. The exact same style of capacity check that *forced* a compromise in prefill (accumulator vs. group reuse) resolves cleanly in decode — same risk pattern, opposite outcome, both real findings, not one being "the right way" and the other "the workaround."
4. A dependency-ordering mistake (assuming SRAM-chunk-size could be solved before lane count) was caught by re-examining Phase 1b's own precedent closely, not by guessing — the discipline of checking "did we actually solve it in this order last time" mattered more than intuition here.
5. Decode doesn't need to inherit every constraint prefill's design faced "for consistency" — the strip-mining insight (§2.5) is a case where blindly re-applying Phase 1b's power-of-2 rule would have cost real efficiency for no reason, once the actual justification for that rule was checked and found not to transfer.
6. `seq_len_q` collapsing to 1 is the single root cause propagating through nearly every hardware-hypothesis divergence from prefill: it broke the systolic array's amortization (§2.1), changed the natural parallelization axis (§2.2), and flipped the accumulator constraint from binding to slack (§2.3) — one workload parameter cascading through the entire design, not a series of unrelated decisions.

---

## 3. Cross-Phase Hardware Comparison: Prefill (1b) vs. Decode (2b)

Direct answer to the spec's own Phase 2 deliverable question: how did the "ideal" hardware differ between prefill and decode, and does that match what you'd expect from the two workloads' arithmetic intensity?

| | Prefill (1b) | Decode (2b) |
|---|---|---|
| Compute primitive | 128×128 systolic array | SIMD/vector engine, 32 lanes |
| Parallelization axis | `d_head` (spatial, shared by both matmuls) | `seq_len_kv` (independent lanes) |
| Dataflow / reuse goal | K/V-stationary across GQA group | Same goal, same reasoning |
| Reuse actually achieved | **No** — accumulator capacity forced `tile_q=32`, sequential per-head processing, ~256× re-fetch | **Yes** — accumulator supports full 8-head concurrency, ~63× margin |
| SRAM-resident chunk | `tile_k=1024` (power-of-2-constrained, ~2× of raw ceiling left unused) | `tile_kv_sram=2043` (strip-mined, ~full raw ceiling used) |
| Dominant bottleneck lever | Fusion (65× AI swing) | None within SDPA except GQA — no fusion lever exists |

**Every one of these differences traces back to the same single root cause: `seq_len_q` collapsing from 8192 to 1** (§1.4, §2.8 Key Finding #6) — not a series of independent design choices. It broke systolic-array amortization (forcing a pivot to SIMD), flipped the natural parallelization axis (the reduction dimension no longer needed to be spatial once the array itself was gone), and flipped the accumulator from binding to slack (the exact same formula, `group_size × seq_len_q × (2+d_head) × 4`, evaluated at `1` instead of up to `32`).

**This matches what the arithmetic intensity numbers predict, precisely**: prefill's hardware is shaped by needing to extract maximum compute throughput from a workload sitting ~17× *above* the ridge point; decode's hardware is shaped by needing to avoid wasting silicon on compute capability the workload structurally cannot use, sitting 30–240× *below* the ridge. The two hardware hypotheses aren't just different in their specifics — they're shaped by opposite design pressures, exactly mirroring the opposite roofline positions found in §1.

---

## 4. Phase 3 — Numerics (Reframed)

Not the spec's literal version (precision-mode throughput comparison on Gemmini) — would both feel generic and hit the same Gemmini-representability wall as 2d (§2.6). Reframed question, hand-derivation only, no tool validation needed (same style as 2a/2b): **is KV-cache quantization (below int8) a bigger lever on decode's AI than GQA was?** Two direct connections motivate this rather than it being a forced addition: (a) Phase 1a explicitly flagged precision as a "carried to Phase 3" open thread (`prefill_notes.md` §1.2 — P's int8 requantization-per-round-trip tradeoff, never resolved); (b) this repo's sibling MoE project already found, for a different severely memory-bound workload, that **numerics — not routing/imbalance — was the dominant lever that actually moved the regime** (top-level `README.md`: "floor scales inversely with dispatch precision, crossover ≈2.5 bytes/element").

### 4.1 The Coupled-vs-Decoupled Precision Fork

**First pass — native low-precision compute**: if K/V is both stored *and* computed natively at half the bit-width (e.g. int8→int4), two things move together: workload bytes halve (AI doubles), but real hardware's peak FLOPs/s also roughly doubles for lower precision — more low-precision ALUs pack into the same silicon budget, the same mechanism behind e.g. a chip's FP8 peak throughput dwarfing its FP32 number. Ridge point (`peak FLOPs/s ÷ peak bandwidth`) doubles too. **Result: `AI/ridge` is exactly unchanged** — `(2×AI)/(2×ridge) = AI/ridge` — no matter how many precision steps down you go. A real, clean cancellation: under this model, numerics is *not* a lever on decode's regime at all, despite genuinely halving bytes moved. This is the same lesson Phase 1c already taught once (ridge point is architecture-specific, not fixed) showing up a second time as *precision*-specific, not just chip-specific.

**Second pass — the realistic case, dequant-before-compute**: real KV-cache quantization decouples storage precision from compute precision — K/V is compressed for storage/bandwidth, then dequantized to a higher precision before the actual dot product (attention scores are numerically sensitive to K/V precision loss). This mirrors a pattern already present in this exact project, applied to a different tensor: Phase 1a's own precision note (`prefill_notes.md` §1.2) — on-chip softmax math runs at higher precision, but P is requantized to int8 only for the HBM round-trip. Under this decoupled model, the compute engine never leaves its baseline precision — **ridge stays fixed at 480.5**. Only bytes move, so the lever is real again, bounded only by how far storage precision can realistically go.

### 4.2 Quantifying the Crossover

Mirrors the MoE project's own `≈2.5 bytes/element` crossover finding. AI scales inversely with bytes/element, holding ridge fixed at 480.5. GQA's current AI ≈ 16 at 1 byte/element (int8). Solving `16 × (1/bytes_new) = 480.5` → **`bytes_new = 16/480.5 = 1/30 ≈ 0.0333 bytes/element ≈ 0.27 bits/element`.**

**Verdict**: not a realistic quantization target by a wide margin — sub-1-bit-per-element isn't a standard precision format, and even the most aggressive real quantization research doesn't approach this for KV caches specifically, given their numerical sensitivity. **Decode attention has a hard, quantization-proof floor**: no realistic (or even wildly unrealistic) K/V precision choice can flip it out of the memory-bound regime.

### 4.3 Cross-Project Synthesis — the Actual Portable Lesson

Striking, direct contrast with the sibling MoE project: there, numerics *was* the dominant, regime-flipping lever, with a real crossover sitting in a realistic range (FP8→BF16, `≈2.5 bytes/element`). Here, the same kind of lever exists structurally (bytes genuinely halve per precision step) but is irrelevant to the regime question.

**The generalizable finding, not specific to either workload**: it isn't "does a numerics lever exist" — it's **how large the roofline margin is that the lever has to close**. MoE's worst-case margin was only `~5–6.7×` (`≈21,065` vs. `≈4,208` ridge, even under imbalance, per the MoE README) — well within what a realistic ~2× byte-format swing can plausibly close. Decode's margin is `30–240×` — an order of magnitude (or two) further out than any believable precision format range could ever reach. Same mechanism, same category of lever, opposite verdict, purely a function of the margin size at the workload's starting point, nothing special about attention vs. MoE routing specifically.

### 4.4 Key Findings — Phase 3

1. Numerics as a lever on regime depends critically on whether compute precision is coupled to storage precision — coupled (native low-precision compute) cancels out via a matching ridge-point shift; decoupled (dequant-before-compute, the realistic case) leaves ridge fixed and the lever intact.
2. Quantified: decode's actual crossover point (`1/30 bytes/element`) is not a realistic target by a wide margin — a clean, hard-floor finding, structurally similar in form to the MoE project's own "hard, imbalance-proof floor" language, but for precision instead of routing skew.
3. **The portable lesson**: whether a numerics lever can flip a regime is a question about the *size of the margin it has to close*, not about whether the lever exists in principle — directly explains why the same technique was decisive for MoE and structurally irrelevant for decode, without needing any difference in mechanism between the two workloads.

---

## 5. Scope: Phase 2c, 2d, 4

**Phase 2c (Timeloop) and 2d (Gemmini/RTL) — both skipped**, full reasoning in §2.6. 2d is structurally blocked (SIMD isn't representable in Gemmini at all). 2c's marginal value was judged low: its real payoff in Phase 1 was tool-methodology lessons already learned once and actively applied here without re-running the tool, against a workload whose conclusions are already more decisive and more thoroughly hand-explained than prefill's were.

**Phase 4 (optional real-hardware check) — skipped.** The "does hand analysis survive contact with reality" theme was already substantively tested twice: once via Phase 1d's actual RTL/Verilator validation (a harder, more rigorous version of what Phase 4 asks for), and once via prior independent work through *How to Scale Your Model* end-to-end. A third pass is diminishing returns on an already-closed theme, and the spec marks this phase optional regardless.

---

## 6. Open Threads Carried Forward

- **Per-lane reduction structure** (§2.2, §2.8): adder tree vs. serial accumulation for each lane's `d_head`-deep dot product — deliberately left unresolved, flagged as unlikely to matter given the roofline margin, not chased further.
- **SRAM-only / no-HBM architectures** (Groq, Cerebras precedent) — flagged mid-derivation as a real, legitimate direction, explicitly out of scope for Gemmini/Timeloop's HBM-based architecture template (same category of tool-representability gap as §2.6). Kept as a footnote for "what I'd explore next" rather than chased further; would need its own ridge-point recomputation under SRAM-scale bandwidth.
- **Sparse/linear attention** — flagged as the natural "next lever after GQA" once GQA became decode's *only* available lever within SDPA (§1.5, Key Finding #6) — explicitly out of scope, a boundary condition worth naming rather than pursuing here.

---

This document covers Phase 2 (2a, 2b, the explicit cross-phase comparison the spec's Phase 2 deliverable calls for) and Phase 3 (numerics, reframed) in full, with 2c/2d/4 treated as deliberate, reasoned scope decisions rather than unfinished work.
