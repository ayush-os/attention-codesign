# Prefill Attention — Workload → Silicon (Phase 1, Complete)

This is the consolidated, polished record of Phase 1 (prefill attention, compute-leaning regime): hand-derived roofline analysis → hardware hypothesis → Timeloop validation → Gemmini RTL validation. It supersedes the working `notes.md` log (Prediction/Log style, kept live during the actual derivation) — this document is organized as a finished writeup rather than a chronological journal, but nothing substantive from that process is lost; self-corrections and wrong turns are kept where they carried real signal, and pure process narration ("status: derived, confirmed" / "next: GQA pass") is cut.

An oral walkthrough of this document (what was learned, what surprised us) is the actual final deliverable for Phase 1, per the project's own choice — this file is the reference it's built from, not a replacement for it.

---

## 0. Workload and Methodology

**Workload** (locked in since Phase 1a, from *How to Scale Your Model*, TPU serving chapter — Llama 3-70B prefill):

| Parameter | Value |
|---|---|
| batch | 32 |
| seq_len | 8192 |
| n_heads (query) | 64 |
| n_kv_heads (GQA) | 8 |
| d_head | 128 |
| d_model | 8192 (= n_heads × d_head; consistent with, not independently re-verified against, the source) |
| precision | int8 throughout (params + KV cache) |

**Methodology** (per the project spec): hand-derive FLOPs/bytes/arithmetic-intensity/ridge-point *before* touching any tool (1a) → hypothesize PE array/dataflow/scratchpad sizing from those numbers (1b) → sweep Timeloop/Accelergy and compare against the hypothesis, explaining every disagreement (1c) → configure Gemmini, read the generated RTL, validate via Verilator, explain every gap (1d). The throughline across all four sub-phases: **gap-hunting — where hand-analysis, simulation, and real hardware disagree, and why — is the highest-value activity**, not getting the "right answer" on the first try.

**Two workload passes were run throughout**, to isolate what GQA actually changes: **MHA** (n_kv_heads = n_heads = 64, a control case) and **GQA** (n_kv_heads = 8, the workload's real configuration).

---

## 1. Phase 1a — Hand-Derived Roofline Analysis

### 1.1 FLOPs

Both matmuls (QK^T and the attention-weights·V product) have the same FLOP count for this shape, via *different* M/N/K assignments — not a shortcut, checked independently for each:

- **QK^T**: per (batch, head), `Q(seq_len, d_head) × K^T(d_head, seq_len)`, contraction dim = `d_head`. FLOPs = `2 × batch × n_heads × seq_len² × d_head`.
- **·V**: per (batch, head), `P(seq_len, seq_len) × V(seq_len, d_head)`, contraction dim = `seq_len` (not `d_head` — the two matmuls contract over different dimensions; their FLOP counts coincide only because `seq_len` appears symmetrically either way for this particular shape).

Softmax FLOPs are negligible relative to either matmul — checked via ratio, not assumed: matmul FLOPs scale with `d_head` (=128) on top of softmax's own `O(seq_len²)` cost, a ~128× gap.

**Total FLOPs = 2 × (2 × 32 × 64 × 8192² × 128) = 2⁴⁶ ≈ 7.037×10¹³** — identical for MHA and GQA. This is the first clean separation in the project: **GQA changes zero compute**. FLOPs are driven purely by query-head count (execution count), not by how many distinct K/V tensors back the computation — KV-head sharing changes *which data* gets reused, never *how many times* the matmul runs.

### 1.2 Bytes Moved

**Compulsory traffic** (Q/K/V/output, read/written once — a *lower bound* assuming perfect on-chip reuse; real mappings may re-read data the scratchpad can't hold, a predicted and later-confirmed source of divergence):

| Tensor | MHA elements | MHA bytes (int8) | GQA elements | GQA bytes |
|---|---|---|---|---|
| Q | 2³¹ | 2 GiB | 2³¹ (unchanged — genuinely per-query-head) | 2 GiB |
| K | 2³¹ | 2 GiB | 2²⁸ (8× smaller — shared across group) | 256 MiB |
| V | 2³¹ | 2 GiB | 2²⁸ | 256 MiB |
| output | 2³¹ | 2 GiB | 2³¹ (unchanged) | 2 GiB |
| **total** | | **8 GiB** | | **4.5 GiB** |

K/V shrink an exact 8× (= n_heads/n_kv_heads). But the **total** compulsory bytes only drop 1.78× (8 GiB → 4.5 GiB), not 8× — because Q and output are structurally invariant to `n_kv_heads` and were already half the MHA total, capping how much the total can improve. **General lesson: an N× reduction in a subterm only produces an N× reduction in the total when that subterm dominates the total.**

**P-matrix (softmax intermediate) traffic** — the dominant lever, and the one that determines regime:

- **Fused** (P never touches HBM — QK^T→softmax→·V pipelined on-chip): additional traffic = 0.
- **Unfused** (P round-trips to HBM 4×: QK^T write, softmax read, softmax write, ·V read): each round trip = `batch × n_heads × seq_len²` elements = 2³⁷ = 128 GiB. Four trips = 2³⁹ = **512 GiB**. Identical for MHA and GQA — P is inherently a per-query-head object, unaffected by KV-head sharing.

*Precision note carried to Phase 3*: on-chip softmax math (max/subtract/exp/sum/divide) runs at higher precision for numerical stability, but P is **requantized to int8 before every HBM write** (and dequantized on read) — preserves the bandwidth rationale for choosing int8 at all, at the cost of added dequant compute and repeated-requantization accuracy loss per round trip.

**Total bytes**:

| | Fused | Unfused |
|---|---|---|
| MHA | 8 GiB | 8 GiB + 512 GiB ≈ **520 GiB** |
| GQA | 4.5 GiB | 4.5 GiB + 512 GiB ≈ **516.5 GiB** (554,587,652,096 B exactly) |

The entire fused-vs-unfused gap (~512 GiB) is P-traffic — fusion, not Q/K/V access, is the dominant bytes-moved lever for this shape.

### 1.3 Arithmetic Intensity and Ridge Point

| | Fused | Unfused |
|---|---|---|
| MHA | 2⁴⁶/2³³ = **8192** | 2⁴⁶/558,345,748,480 ≈ **126** |
| GQA | 2⁴⁶/4.5 GiB ≈ **14,564** | 2⁴⁶/554,587,652,096 ≈ **126.9** |

Both exact ratios are structural, not coincidental: fused/unfused = 65× (mirrors the 2³³(1+2⁶) bytes factorization); GQA/MHA fused = 16/9× (mirrors the 8 GiB/4.5 GiB bytes ratio).

**Ridge point**, derived from the workload's own source chip for internal consistency (TPU v5e, int8): peak FLOPs/s = 3.94×10¹⁴, HBM BW = 8.2×10¹¹ B/s → **C ≈ 480.5 FLOPs/byte** (same order of magnitude as Pope et al.'s "~300" rule of thumb, not identical — consistent with that being a different-chip-generation heuristic, not a universal constant).

### 1.4 Conclusion and Key Takeaways

**Fused AI (8192, 14564) is decisively above the ridge (480.5); unfused AI (126, 126.9) is decisively below it — not a close call in either direction.** This workload is not intrinsically compute- or memory-bound: **the regime is determined almost entirely by one implementation decision — does the softmax/P intermediate ever touch HBM — not by workload shape.**

1. **Regime is a design decision, not a workload property.** Same shape, opposite conclusion, depending purely on fusion.
2. **FLOPs are blind to KV-head organization; bytes are not.** A clean separation of "compute lever" (shape/algorithm) from "memory lever" (data organization/fusion).
3. **GQA's payoff is regime-dependent (Amdahl's Law).** Unfused: K/V is ~1% of total bytes (P-traffic dominates) → the real 8× local win produces ~0% total win. Fused: K/V was co-equal with Q/output → the same 8× local win becomes a real 16/9× total-AI win. Same technique, same local speedup, opposite payoff — purely a function of what fraction of the *current* bottleneck it touches.
4. **Fix the dominant bottleneck before chasing secondary optimizations.** Fusion (65× AI swing) dominates GQA (second-order unless fusion is already solved) — a design-priority ordering, not just an arithmetic curiosity, that generalizes to any later stack of optimizations (e.g. Phase 3 precision on top of dataflow decisions).
5. **Methodological habits that paid off**: computing both fused and unfused bounds instead of picking one (surfaced the fusion/GQA interaction, which a single-path analysis would have hidden); deriving the ridge point from the workload's own source chip instead of a remembered heuristic, then cross-checking against the heuristic anyway.
6. **Open thread into 1b**: every bytes-moved number assumes "perfect on-chip reuse" — and that assumption is more demanding for GQA (needs one K/V head resident across *8* query heads' worth of work, not 1). Whether that's realistic for a real scratchpad is deferred to 1b's sizing exercise.
7. **GQA's benefit is regime-dependent in a second, deeper way.** Roofline time = `max(FLOPs/peak_compute, bytes/peak_bandwidth)`. In **fused prefill** (decisively compute-bound), the FLOPs term sets execution time, and GQA leaves FLOPs completely unchanged — so **GQA's bytes reduction does not reduce fused-prefill execution time at all**, despite improving the AI number. Its real prefill payoff is scratchpad/KV-cache pressure, not throughput. The throughput win is where decode (Phase 2, memory-bound by nature) should show a genuinely different kind of benefit from the same technique — direct material for the Phase 1-vs-Phase 2 comparison.

---

## 2. Phase 1b — Hardware Hypothesis (PE Array, Dataflow, Scratchpad Sizing)

Goal per spec: a *defensible* hypothesis to test against Timeloop, not a guaranteed-correct one.

### 2.1 PE Array Shape

Initial instinct — size the array to the full GEMM dims (8192×8192) — was rejected after checking real precedent: TPU MXU 256×256, Trainium 128×128, Nvidia tensor cores ~16×16, all orders of magnitude below `seq_len`. **Array size is a hardware design choice, independent of workload scale**; `seq_len` gets tiled through a much smaller array instead.

`d_head` (=128) is a strong candidate for one array axis: native to Q/K/V's own tensor shape (not scale-dependent like `seq_len`), and appears in *both* matmuls — as the contraction dim (K) in QK^T, and the output dim (N) in ·V.

**Resolved via first-principles systolic-array framework**: a systolic array makes exactly 2 of a GEMM's (M,N,K) dims spatial at once; the dataflow name (weight-/output-/input-stationary) *is* the choice of which pair, and the 3rd dimension streams/accumulates temporally regardless of size. Under weight-stationary with K/V as the stationary operand: QK^T → spatial = (`d_head`, k-tile), temporal = `seq_len_q`; ·V → spatial = (k-tile, `d_head`), temporal = `seq_len_q`. **Both matmuls need the identical spatial pair, just transposed** — a direct structural consequence of `d_head` being the one dimension shared by every operand under weight-stationary, discovered by working out each matmul's spatial/temporal split independently and noticing they matched (not assumed in advance). This resolves the cross-matmul utilization question — **given** the array/feed logic can route either dimension onto either physical axis between phases, a stated assumption at the time, later confirmed in Phase 1d (§4.5).

No exact area/power budget exists to pick the second axis definitively, and array width is itself a Timeloop sweep dimension — **locked in 128×128 as the primary hypothesis** (exact Trainium precedent), **128×256 as an explicit alternate** (roughly halves passes through the scratchpad-resident k-tile at ~2× array cost; no new utilization penalty found at either size).

### 2.2 Dataflow

"Stationary" defined concretely: which GEMM dimension sits *spatially* on the array (loaded once, fixed across cycles) vs. streams *temporally*.

- **MHA**: no cross-head reuse for Q or K (every head is unique data) — dataflow choice doesn't matter for MHA from a reuse standpoint.
- **GQA**: K (and V) are reused across the group of `n_heads/n_kv_heads` = 8 query heads sharing one KV head. **Concluded: K/V stationary, Q streamed** — directly the same 8× reuse factor that drove the Phase 1a GQA byte savings, not separately derived. (An initial inversion — "streaming K/V saves traffic" — was self-corrected once tied back to the mechanism: reuse *without* re-fetching is stationary, not streaming; Q being the larger tensor also argues against making it stationary, a second independent angle to the same conclusion.)

### 2.3 Scratchpad and Accumulator Sizing

Distinguished array-level "stationary" (holds for one tile-pass) from scratchpad-level residency (must span the *entire* 8-head group for GQA's compulsory-byte claim to actually hold) — conflating these was an early, caught mistake. This "stationary is scoped to a specific memory boundary" pattern recurred three times in this project: scratchpad-vs-accumulator, array-vs-scratchpad (below), and HBM-vs-scratchpad (K/V "stationary" means HBM isn't re-touched per head, not that the array's own PE contents are frozen).

**Real Gemmini defaults** (verified via GitHub/paper, not assumed): 256 KB scratchpad (up to ~1 MiB across banks) + 256 KB accumulator, as two *separate* physical memories — adopted as the sizing target instead of TPU-scale SRAM (rejected two tempting-but-wrong comparisons along the way: TPU 8t/8i's 128/384 MiB on-chip SRAM, and TPU v1's 4 MB accumulator — both real, verified facts, but the wrong comparison basis, a custom datacenter ASIC vs. a small RTL generator).

A full P tile (`seq_len_q × seq_len_k`) = 64 MiB per head — far too large for any real scratchpad, so **achieving "fused" forces tiling of `seq_len_k` (and `seq_len_q`) regardless of GQA** — the P-size-driven and GQA-reuse-driven tiling requirements compose rather than conflict. Preserving GQA's chunk reuse under tiling requires holding one K/V chunk fixed while sweeping all 8 heads in the group — which requires per-head **online-softmax** tracking state carried across the chunk sweep. This mechanism (running max, running sum, rescale-on-update) was **independently reconstructed from first principles** (pure memory-traffic-constraint reasoning) before being told the Flash Attention name for it — later validated directly against real Gemmini hardware in Phase 1d (§4.4).

**Solving for `tile_q`** (accumulator, ≤256 KiB): per-head state = running max + running sum + partial-output accumulator (`tile_q · (2 + d_head) · 4` bytes at fp32 — the `d_head`-scaled output-accumulator term dominates). Group size scaling this is `n_heads/n_kv_heads` = 8 (coincidentally equal to `n_kv_heads` for this shape — the formula must reference group size, not `n_kv_heads` directly, to generalize). `8 × 520·tile_q ≤ 262,144 B` → max ≈ 63 → **tile_q = 32** (hardware-friendly power of 2).

A second, initially-missed accumulator term was caught while cross-checking against the 128×256 alternate array (which broke a coincidental size-match that had hidden it): a transient **raw S/P block** (pre-softmax, higher-precision) needs accumulator residency before requantization to the int8 P that lives in scratchpad — distinct from the partial-output accumulator, sized `tile_q × 128` (~16 KB, one reused buffer, not ×8 heads).

**Softmax granularity**, resolved fine-grained (not the original coarse hypothesis): coarse (one softmax update per full `tile_k` chunk) was chosen first to minimize control instructions, but once the raw-S term above was honestly counted, coarse *overflows* the accumulator budget by 2 KB (133,120 B running state + 131,072 B coarse raw-S = 264,192 B > 262,144 B). **Fine-grained** (per 128-wide array sub-pass) fits with large margin (16 KB vs. 128 KB) at the cost of 8× more control instructions — also converges toward the actual Flash Attention structure rather than an ad-hoc alternative.

**Solving for `tile_k`** (scratchpad, ≤1 MiB): double-buffered K/V (`512·tile_k` B) + fixed fine-grained P (4,096 B, no longer scaling with `tile_k`) + fixed Q/output (8,192 B) ≤ 1,048,576 B → ceiling ≈ 2,024. But `seq_len_k` = 8192 = 2¹³ means only power-of-2 tile sizes divide it evenly (no ragged final chunk), and 2048 exceeds the ceiling — **`tile_k` = 1024 stands**, now justified by workload-shape divisibility rather than a "feels hardware-friendly" heuristic.

Accumulator capacity beyond the 256 KB default was logged as a genuine free sweep parameter for Phase 1c (128 KB–512 KB), the 512 KB endpoint grounded in a real, benchmarked Gemmini "BigSP" config (512 KB scratchpad + 512 KB accumulator + 1 MB L2) — with a calibrating caution attached: that same published figure's measured speedup for BigSP on the **Matmul** benchmark category (closest analog to attention) was small (~1–3%), versus **Conv**'s ~10–11% — a real data point against assuming "bigger accumulator" proportionally fixes anything.

### 2.4 Final Hypothesis

- **PE array**: 128×128 primary (Trainium precedent), 128×256 explicit alternate.
- **Dataflow**: weight-stationary, K/V as the stationary operand (8× GQA reuse factor), Q streamed; one physical array serves both matmuls via `d_head`'s shared spatial role (given the axis-routing assumption, later confirmed).
- **Scratchpad** (≤1 MiB): double-buffered K/V chunks (`tile_k`=1024) + fixed fine-grained P tile (`tile_q×128`) + Q tile + output tile.
- **Accumulator** (≤256 KiB, 512 KiB flagged as a real alternate): per-head online-softmax state for the 8-head group (`tile_q`=32) + transient raw-S/P block (~16 KB, not ×8).
- **Softmax granularity**: fine-grained (per 128-wide sub-pass). Executes on a separate vector/scalar unit, not the systolic array (array idle throughout) — whether Gemmini has a native unit for this or needs the host core was left as an explicit open Phase 1d question (resolved in §4.4).
- **Stated assumptions carried forward**: (a) array/feed logic can route either GEMM dimension onto either physical axis between phases [confirmed, §4.5]; (b) fp32 online-softmax state (revisit at fp16 in Phase 3); (c) K/V double-buffered, Q/output not; (d) fine-grained softmax; (e) softmax's execution unit unspecified [resolved, §4.4].

### 2.5 Major Open Finding: K/V Reuse vs. Accumulator Capacity (Loop-Order Tension)

Walking the *full* loop nest explicitly (batch, KV-group, Q-tile, K/V-chunk, head, array sub-pass) — not reasoning about each level in isolation — surfaced a real tension: `tile_q`=32 was sized against **one Q-tile's** worth of simultaneous 8-head accumulator state; there's no budget for holding multiple Q-tiles concurrently. This forces **Q-tile to be the outer loop relative to K/V-chunk** (must finish one Q-tile, cycling all 8 chunks, before starting the next) — which means **each K/V chunk gets re-fetched from HBM ~256× (once per Q-tile) instead of once per group**, breaking the "fetched once per group" GQA reuse assumption underlying Phase 1a's compulsory-bytes lower bound.

This is not a broken hypothesis — it's a concrete, hand-derived instance of exactly the category of gap Phase 1a's own bytes-moved section predicted in advance ("real mappings may re-read data if scratchpad can't hold what's needed"). All FLOPs work, the roofline/AI methodology, ridge-point derivation, and dataflow/array-shape derivation are entirely unaffected — this only concerns whether *this specific* `tile_q`=32 config achieves full vs. partial K/V byte-savings. A real, unexplored lever exists (a smaller `tile_q` trades more control instructions for holding multiple Q-tiles, directly reducing re-fetch frequency) but was **deliberately left for Timeloop to resolve** rather than hand-optimized further, per the project's "defensible hypothesis, not optimal" framing.

### 2.6 Key Takeaways

1. **Array sizing is governed by real hardware precedent and dataflow-driven reuse, not workload scale.** `seq_len` never appears in the array's shape at any level — tiled away twice (to a scratchpad-resident chunk, then to an array-sized sub-tile).
2. **Dataflow and array shape aren't separable questions.** Array shape has no answer until the stationary operand is fixed, since dataflow determines which GEMM dimensions even compete for the two physical axes.
3. **`d_head`'s presence in every operand is *why* one array serves both matmuls** — a structural property of attention's shape, discovered by working out each matmul independently and noticing the match.
4. **Scratchpad and accumulator are separate physical resources with different natural occupants**, mirroring the Phase 1a precision decision onto physical memory placement.
5. **Tiling is a two-level phenomenon, easy to conflate**: workload scale → scratchpad-resident chunk (one budget) → array-sized sub-tile (a second, independent budget).
6. **When a tradeoff can't be solved exactly, state a primary hypothesis plus an explicit alternate** rather than forcing one answer — especially when the next tool (Timeloop) sweeps exactly that dimension anyway.
7. **GQA's benefit is regime-dependent in two distinct ways**, spanning 1a and 1b: Amdahl's-Law-style (fused vs. unfused bytes) and roofline-position-style (fused prefill's compute-bound time isn't helped by GQA at all — its real payoff is scratchpad/KV-cache pressure).
8. **"Stationary" is always scoped to a specific memory boundary** — recognized independently three times (scratchpad-vs-accumulator, array-vs-scratchpad, HBM-vs-scratchpad).
9. **Raw matmul output and post-processing output are different objects, even when they coincidentally match in size** — the pre-softmax S block and post-softmax P looked identical only because the primary array hypothesis is square; the 128×256 alternate exposed them as independent terms.
10. **A choice made for one reason can silently violate a different, unrelated constraint** — coarse softmax granularity was chosen to minimize instructions, but broke the accumulator budget once the raw-S term was honestly counted.
11. **Workload shape can turn a "hardware-friendliness heuristic" into a hard requirement** — `seq_len_k`=2¹³ means `tile_k`=1024 was the *only* valid power-of-2 choice, not a style preference.
12. **Major finding**: capturing GQA's cross-head reuse and having enough per-head Q-tile capacity are in direct tension under a small, fixed accumulator — the mechanism that lets you exploit the reuse is what starves the capacity needed to avoid re-fetching. Not a sign GQA "doesn't work" — a sign this specific config likely doesn't realize the full theoretical 8× byte savings.
13. **When you hit a wall like #12, check whether it's fundamental or an artifact of one specific choice.** Here it's the latter — a real, deliberately-unresolved lever left for Timeloop.
14. **Cross-chip comparisons need to match constraint regime, not just be factually real** — TPU v1's 4 MB accumulator is real, but citing it for a small RTL generator's sizing is the wrong comparison basis, twice repeated (TPU 8t/8i, then TPU v1).
15. **The right resolution was checking the actual target tool's own documented flexibility** (Gemmini's `acc_capacity` as a real, user-configurable parameter, benchmarked in a published "BigSP" config) rather than reaching for an unrelated chip.
16. **Real data can be calibrating, not just confirming** — BigSP's small (~1–3%) Matmul-category speedup is a genuine caution against assuming "bigger accumulator" proportionally fixes the re-fetch tension.
17. **How 1b and 1c divide labor**: 1b produces specific, justified predictions wherever hand-derivation reaches one, and explicitly flags (not forces) the rest. 1c is a genuine search with hand-derived numbers as falsifiable comparison points, not a hand-curated shortlist to pick from.

---

## 3. Phase 1c — Timeloop/Accelergy Validation

Ran in a separate repo/Docker environment (`timeloop-accelergy-exercises/workspace/attention_1c/`) — QK^T and ·V expressed as conv-style GEMM problems, an architecture translating the Phase 1b hypothesis into a sweepable Timeloop model (128×128 array, 1 MiB scratchpad / 256 KiB accumulator, DRAM at TPU v5e HBM bandwidth), **dataflow left unconstrained at the scratchpad** so the mapper had to discover K/V-stationary on its own rather than being told.

*(This phase was run end-to-end in a since-destroyed session — a repo rename killed the chat mid-run. Tool artifacts and the Docker container survived on disk and were fully recovered; nothing was lost, but the recovery process is why this section reads as a factual reconstruction in places.)*

### 3.1 Results

| Run | Clock model | Kernel | Utilization | Cycles | Winning scratchpad-resident dataspace |
|---|---|---|---|---|---|
| `primary_qkt` | 1 GHz | QK^T | **100%** | 1,073,741,824 (2³⁰, compute-bound ideal) | none |
| `primary_qkt_v2` | 2 GHz, int8-pumped (DRAM BW held fixed in bytes/s) | QK^T | **100%** | 1,073,741,824 | none |
| `primary_v` | 1 GHz | ·V | **100%** | 1,073,741,824 | output-stationary |
| `primary_v_v2` | 2 GHz corrected | ·V | 80.63% | 1,331,701,751 | weight-stationary (V-stationary — matches the Phase 1b hypothesis), but a worse local optimum than v1's OS result |

DRAM traffic confirms this setup models the **unfused** regime exactly: `primary_qkt`'s winning map shows `Weights(K)`=256 MiB (matches Phase 1a's GQA K bytes exactly) + `Inputs(Q)`=2 GiB + `Outputs(P)`=128 GiB (one full P write) — and `primary_v`'s `Inputs` shows the identical 128 GiB read back in.

`_v3` runs (same v2 architecture, much larger search budget) ran **over 2 days** before being killed — 100% utilization was already the theoretical ceiling and already found; remaining runtime could only chase marginal energy improvements at the same cycle count, not a different answer.

### 3.2 Four Findings

**1. Ridge point is a property of the specific accelerator being modeled, not a fixed workload characteristic.** QK^T hits 100% utilization even under the full unfused 128 GiB P-write — looks like it contradicts Phase 1a's "unfused should be memory-bound" prediction (AI≈126 vs. ridge 480.5), but this Timeloop architecture models a **single** 128×128 array at 1 GHz (not TPU v5e's 4 MXUs at 1.5 GHz) — 6× lower peak compute (4× from array count, 1.5× from clock) at the same real HBM bandwidth. This architecture's *own* ridge point is 480.5/6 ≈ **80.08**, not 480.5. AI≈126 clears 80.08 — compute-bound was correct all along for this specific modeled system. **Generalizes**: any Phase 1a/1b roofline conclusion checked against a Phase 1c/1d tool result needs its ridge point recomputed for the actual modeled hardware — extending Phase 1a's "regime is a design decision" theme one level further: regime is also relative to *which hardware* you're asking about.

**2. How much dataflow choice matters is itself regime-dependent.** ·V's winning dataflow differed qualitatively between v1 (output-stationary, 100%) and v2 (weight-stationary, 80.63%) even though only the clock/bandwidth model changed. Confirmed via the *identical mapping* appearing in both raw search logs: 100% under v1, 80.63% under v2 — a real DMA/compute-overlap effect (this schedule's DMA time hid fully behind compute under v1's clock, and stopped hiding once compute got 2× faster against the same fixed DRAM bandwidth), not noise. Near/below an architecture's own ridge, competing dataflows were essentially fungible (WS/OS tied within a rounding error in v1); once the ridge doubled, the same mapping fell far behind and which dataflow "wins" started to matter a great deal more. Verified v2's true ceiling is still 100% (the killed `_v3` log already showed multiple 100%-utilization samples under v2's architecture) — 80.63% is a local optimum, not the ceiling.

**3. A single mapper run's "winner" is not necessarily the true optimum — it's whatever a deterministic, budget-limited search happens to find.** Rerunning `primary_v_v2`'s exact config produced a **byte-for-byte identical** search log and the same 80.63% answer — `random_pruned` is deterministic given a fixed config, not randomly seeded per invocation. Escaping the local optimum needs materially more search depth, not another attempt at the same depth — a methodology lesson for interpreting *any* Timeloop mapper result, not just this one.

**4. This Phase 1c setup cannot model "fused" at all, structurally — not a mapper/architecture finding.** Every run's DRAM traffic shows the full 128 GiB P round-trip regardless of dataflow, because QK^T and ·V are two independent Timeloop problems with no on-chip path connecting one's output to the next's input — no mapper search, however deep, can change that. This matters because **Phase 1b's entire scratchpad/accumulator sizing exercise was designed specifically to enable fusion** — meaning Phase 1c has only ever characterized the *unfused* regime. Decision made: not worth hacking Timeloop to fake fusion (its problem format isn't built for two matmuls sharing an intermediate that never leaves the chip) — real fusion validation deferred to Phase 1d, where a Gemmini RTL pipeline can literally keep P in scratchpad between the two matmuls, no modeling hack required. (An unexplored footnote: a cheap hand-projected "fused-equivalent" estimate is possible — subtract shared P-traffic bytes/energy from the existing unfused numbers using Phase 1a's own byte formulas — not computed, not needed once Phase 1d's real fusion behavior was in hand.)

### 3.3 Key Takeaways

1. Ridge point is architecture-specific, not carried over from a workload's reference chip — generalizes Phase 1a's "regime is a design decision" theme to "regime is also relative to which hardware you're asking about."
2. Dataflow sensitivity is itself regime-dependent — near an architecture's own ridge, dataflow choice barely matters; the more headroom above the ridge, the more it matters.
3. A mapper's single-run "winner" is a local optimum specific to its search depth, not a ground truth — the same tool can report a worse answer for a materially better architecture if the search never gets deep enough to find the reachable ceiling.
4. A tool's structural modeling limitations can silently scope out the exact regime a hardware hypothesis was built to test — worth checking, for any tool, whether its native problem representation can even express the hypothesis before trusting results as validation of it.

---

## 4. Phase 1d — Gemmini Configuration and RTL Validation

### 4.1 Scope Decision

Phase 1c's signal-to-time ratio was judged poor in retrospect — days of wall-clock for findings that read more as Timeloop-tool-methodology lessons than new codesign insight, versus 1a/1b's much higher learning density. A back-of-envelope check (2⁴⁶ FLOPs over a 128×128 array's peak MAC throughput, at Verilator's realistic kHz-range simulation speed) confirmed that running anything near the real Phase 1a workload scale through Verilator is genuinely **multi-day**, not just "feels slow." Gemmini's own software stack also has no stock attention kernel — building one would be real kernel-engineering effort the spec never actually required (the spec's own wording: "your attention kernel *or a representative slice of it*").

**Decision: a scoped-down 1d.** Kept: a Gemmini config matched to the Phase 1b/1c hypothesis, real RTL generated and read, real Verilator execution on stock tests, resolution of the softmax-unit and axis-routing open questions. Dropped: a custom attention kernel, any real-workload-scale run. **Consequence, stated explicitly rather than glossed over**: the Phase 1 "gap analysis" here is qualitative/structural (RTL literacy + open-question resolution), not a rigorous quantitative Timeloop-vs-real-cycles comparison.

### 4.2 Environment

Chipyard/Gemmini/Verilator ran on **Stanford Farmshare**, not the local machine (Apple M3 Max, 36 GB unified memory) — not a compute/RAM constraint (36 GB is plenty for a single-config Verilator sim), but toolchain compatibility: Chipyard's install scripts, RISC-V GNU toolchain prebuilts, and Verilator versions target Linux x86_64, not macOS/ARM64. Timeloop/Accelergy (lighter-weight, Docker-based) ran fine locally and stayed there — only the Chipyard/Gemmini/Verilator toolchain needed the Linux environment.

### 4.3 Config Chosen: `AttentionPrefillRocketConfig`

Array scaled down from the Phase 1b/1c 128×128 hypothesis to **32×32** — matches `largeChipConfig`, the largest real precedent in Gemmini's own `Configs.scala` (no 128×128 precedent exists there, and 16×16→128×128 is a 64× PE-count jump in fully-unrolled combinational Chisel hardware, a real elaboration-time risk with no payoff once the RTL-vs-Timeloop comparison was already descoped to qualitative). Added as a new, additive config rather than editing `defaultConfig` in place:

- `meshRows = meshColumns = 32`, `tileRows = tileColumns = 1` → 32×32 array.
- `dataflow = Dataflow.WS` — weight-stationary only, matching the Phase 1b hypothesis.
- `sp_capacity` ≈ 1 MiB, `acc_capacity` = 256 KB — matching Phase 1b/1c sizing.
- Based on `defaultConfig.copy(...)`, not `chipConfig`/`largeChipConfig`, to keep `ex_read_from_acc`/`ex_write_to_spad` at their permissive defaults (those two disable read/write paths as tapeout-area optimizations, not wanted here).

Lives in `generators/gemmini/src/main/scala/gemmini/Configs.scala` (+ `AttentionPrefillGemminiConfig` mixin) and `generators/gemmini/chipyard/GemminiConfigs.scala` (top-level SoC config, alongside the existing `GemminiRocketConfig` etc. — nothing existing was touched).

### 4.4 Finding 1 — WS-Only Hardware Behaves Correctly (Functional Validation)

Stock `matmul-baremetal` (built from `bareMetalC/matmul.c`) **failed** on the new config (`*** FAILED ***` assertion, real RTL executed to completion, output ≠ golden reference). Root-caused rather than retried blind: `matmul.c`'s own header comment states its purpose is explicitly to "check whether we can switch between output- and weight-stationary dataflows" — its loop sweeps `dataflow` from `OUTPUT_STATIONARY` to `WEIGHT_STATIONARY`. Our config's `Dataflow.WS` compiles out OS support entirely, so the OS-mode loop iteration issues a request the hardware cannot execute correctly — wholesale-wrong output, consistent with a full dataflow-mode mismatch rather than a numerical bug.

Reran with `matmul_ws-baremetal` (the WS-only stock variant): **passed** (exit code 0, clean `$finish`, no assertion). First concrete, mechanistically-explained Phase 1d result, obtained without a custom kernel or full-workload run.

*(Incidental toolchain finding, not conceptual: a `make` invocation started before `env.sh` is sourced can bake a broken `$RISCV`-derived include path into generated Makefiles, and fixing the shell's env afterward doesn't force regeneration — needs a clean `rm -rf` of that config's `generated-src`/`simulator-*` and a fresh build once the environment is correct from the start.)*

### 4.5 Finding 2 — Gemmini Has a Native Softmax Hardware Unit

Resolved via Gemmini's own source (`Activation.scala`, `Normalizer.scala`, `dev` branch) — a factual/reference lookup, not a derivation:

- `Activation.scala` defines a dedicated `SOFTMAX` activation-function code, alongside `RELU`, `LAYERNORM`, `IGELU`.
- The `Normalizer` module maintains **running max** and **running sum** registers, and computes `iexp(d - max, ...)` — a hardware integer/quantized exponential approximation, explicitly max-subtracted before exponentiating (the standard numerically-stable softmax trick). Final normalization uses a hardfloat divider computing `1/sum`.
- **Resolves the open Phase 1b question**: softmax does *not* need to route through the host Rocket/BOOM core — Gemmini has a native vector/scalar-style unit for it.
- **A genuine validation, not just a resolved unknown**: this running-max/running-sum/max-subtracted-exp structure is essentially the same online-softmax mechanism independently derived by hand in §2.3, before knowing the Flash Attention name for it. Real Gemmini hardware designers converged on the same mechanism, for the same numerical-stability reason.
- **Still open, not chased further**: whether this unit's granularity matches the specific fine-grained, `tile_q`=32 tiling scheme hypothesized in Phase 1b, or assumes a simpler single-pass case — would need deeper reading of `AccumulatorScale.scala`/the ISA's norm-command sequencing. Judged lower payoff than the axis-routing question (§4.6) given the project's own "know when to stop" instinct; left as an explicit footnote.

### 4.6 Finding 3 — What `dataflow = Dataflow.WS` Actually Changes at the RTL Level

Read the generated RTL for `AttentionPrefillRocketConfig` directly, cross-referenced against Gemmini's Chisel source when generated-Verilog naming alone became inconclusive.

- **Bit widths confirmed as predicted**: the real systolic PE's port list shows 8-bit inputs (`inputType = SInt(8.W)`) and 20-bit intermediates (`spatialArrayOutputType = SInt(20.W)`) — validated directly against generated signal declarations.
- **Naming gotcha, caught before a wrong conclusion**: the "obviously" named `PE.sv` is actually generated from `Transposer.scala` — an unrelated module that happens to also be named `PE` in Chisel source. The *real* systolic PE (from `PE.scala`) got Chisel/firtool-uniquified to `PE_1024` to avoid the collision — the "1024" suffix is a disambiguation artifact, **not** a literal encoding of 32×32 PE count (confirmed: the `Tile` module instantiates exactly one `PE_1024`, consistent with `tileRows=tileColumns=1`).
- **Central finding**: the real PE's port list still includes a live `io_in_control_dataflow` input — a generic OS/WS-select wire — even though the config sets WS-only. Tracing the enforcement mechanism through `ExecuteController.sv` hit a real dead end in pure Verilog-naming forensics (the enqueue-side connection for this field was untraceable by name — several grep dead-ends, informative in aggregate but not conclusive alone) — resolved instead by reading the actual Chisel source directly:
  ```scala
  val current_dataflow = if (dataflow == Dataflow.BOTH) Reg(UInt(1.W))
                          else dataflow.id.U
  ```
  Since the config sets `Dataflow.WS` (not `BOTH`), this is a **compile-time Chisel literal constant**, not a runtime-writable register. Only a `Dataflow.BOTH` config generates a real register, written at runtime by decoding a `CONFIG_EX` instruction.
- **Revises the Phase 1b hardware-cost framing**: `PE.scala` itself is generic regardless of the `dataflow` parameter — WS-only is enforced by wiring the PE's existing generic control port to a fixed constant at elaboration time, **not** by generating a structurally smaller/simpler PE. The OS-capable logic is still fully present in this RTL; whether a later synthesis pass (not reached here — this project stopped at RTL/Verilator) would dead-code-eliminate it is a separate, unconfirmed question. The config choice is validated correct at the *instruction/software* level (matches the §4.4 pass/fail result exactly) — the *area/power* savings implied by "WS-only, driven by GQA reuse" in the original Phase 1b hypothesis are not established at this level. A real, honest gap between the hypothesis's framing and what this generator actually does.

### 4.7 Finding 4 — Axis-Routing Assumption Confirmed (Gemmini's Transposer)

Phase 1b (§2.1) carried an explicit, unverified assumption: the array/feed logic can route either GEMM dimension onto either physical axis between the QK^T and ·V phases — needed so one physical array serves both matmuls. Verified directly via Gemmini source (`MeshWithDelays.scala`) and docs: Gemmini has a real hardware **transposer module**, gated by software-visible `a_transpose`/`bd_transpose` instruction flags, with routing logic dependent on both the flag and the active dataflow mode:

```scala
a_is_from_transposer = Mux(dataflow === Dataflow.OS, !a_transpose, a_transpose)
b_is_from_transposer = dataflow === Dataflow.OS && bd_transpose
d_is_from_transposer = dataflow === Dataflow.WS && bd_transpose
```

The concrete, non-obvious reason this exists at all (from Gemmini's own docs): in output-stationary mode the transposer is used **even when software doesn't request it**, because a matrix's rows are stored contiguously within one scratchpad SRAM row, but the array needs same-row elements entering the *same* PE sequentially rather than adjacent PEs simultaneously — a transpose has to happen purely as a consequence of dataflow mode, independent of programmer intent.

**Resolution**: the Phase 1b assumption is confirmed by a real hardware mechanism, not just shown plausible — and for a reason (SRAM row layout vs. per-PE feed order) that has nothing to do with attention specifically. A small closed loop: the `PE.sv` naming-collision confusion in §4.6 turns out to be this exact same transposer module.

**Scope caveat**: this confirms the *capability exists* in the generator/ISA. It does not confirm a real QK^T→·V kernel would correctly sequence the transpose flags across the two phases — that would only be verified by actually writing the custom attention kernel, deliberately descoped per §4.1.

### 4.8 Key Takeaways

1. **A hardware generator's config knobs aren't uniform in what they change, and the config schema alone can't tell you which kind a given knob is.** `meshRows`/`meshColumns` are elaboration-time/structural. `dataflow` is behavioral/control-constant — same generic PE, same port, just a fixed-vs-runtime source for one control wire. Indistinguishable from the config file alone; only tracing each through the actual generator source reveals which is which.
2. **"Restricting" a dataflow parameter doesn't necessarily buy the area/power savings naturally assumed from "removing support for a mode."** Real savings, if any, depend on a synthesis-time dead-code-elimination step this project didn't reach — revises, not overturns, the Phase 1b hardware-cost framing.
3. **Functional pass/fail testing proves *that* a restriction is enforced, not *where* or *how*.** Neither the software-level test result nor the RTL/source reading alone would have been sufficient; both phases of investigation were necessary.
4. **Pure grep/naming archaeology on generated Verilog has a real ceiling — knowing when to stop and cross-reference the actual HDL source is itself a skill.** Both the naming-collision confusion and several inconclusive grep dead-ends cost real time before going straight to the source settled each question in one lookup.
5. **The Normalizer/softmax finding is a genuine validation, not just a resolved unknown** — real hardware designers independently converged on the same mechanism hand-derived from first principles in Phase 1b.
6. **A "stated assumption, not yet verified" from an earlier phase can turn out to be backed by real, purpose-built hardware — for a reason neither hand-derivation nor Timeloop's cost model would surface.** The axis-routing mechanism exists because of a scratchpad-SRAM-layout constraint, not because of anything about attention or the QK^T/·V pairing. Also: confirming a capability exists in the generator/ISA is not the same as confirming a real kernel invokes it correctly — that gap stays open by design.

---

## 5. Cross-Phase Synthesis

**What generalizes across all four sub-phases:**

- **Regime (compute- vs. memory-bound) is never a fixed workload property** — it's determined by an implementation decision (fusion, Phase 1a), then further shown to be a property of *which specific hardware* you're asking about (ridge point recomputation, Phase 1c), and finally shown to have a software/hardware split even within one hardware config (dataflow-as-constant, Phase 1d). The same theme sharpens at every level of the stack.
- **Gap-hunting was in fact the highest-value activity**, exactly as the spec claimed — every phase's most useful output was a place where hand-analysis, a tool, or real hardware disagreed with the working hypothesis, not a place where they agreed. Four independent, mechanistically-resolved findings in 1c; at least four more in the scoped-down 1d.
- **A tool's (or a generator's) native representation can silently fail to express the thing you built a hypothesis to test** — Timeloop's two-independent-problems structure can't model fusion at all (1c); a config parameter can look like it changes hardware structure when it actually just changes a control constant (1d). Worth checking, for any tool, whether it can even express the hypothesis before trusting its results as validation.
- **Predicting a gap in advance and then finding a concrete instance of it by hand is the prediction paying off, not a setback** — the K/V-reuse-vs-`tile_q` tension (Phase 1b) is a direct, hand-derivable instance of exactly the divergence-category Phase 1a's own bytes-moved section predicted.
- **Independently re-deriving a known technique from first principles, then finding real hardware that implements the same mechanism, is strong evidence the derivation was sound** — not a coincidence to shrug off (online-softmax / Flash Attention, confirmed against Gemmini's real `Normalizer` unit).
- **Scoping decisions made under real time constraints are legitimate methodology, not corner-cutting — as long as they're stated honestly.** The 1d scope-down (no custom kernel, no full-workload run) traded a rigorous quantitative RTL-vs-Timeloop comparison for a cheaper qualitative one, and got real, load-bearing findings anyway (Findings 1–4, §4.4–4.7).

---

## 6. Open Threads Carried Forward

- **Softmax-unit granularity** (§4.5): does Gemmini's native `SOFTMAX` unit match the fine-grained, `tile_q`=32 online-tiling scheme, or assume a simpler single-pass case? Flagged, not chased.
- **Real-kernel transpose sequencing** (§4.7): the axis-routing *capability* is confirmed; whether a real QK^T→·V kernel correctly sequences `a_transpose`/`bd_transpose` across the handoff was never checked (would require writing the custom kernel, deliberately descoped).
- **Synthesis-level area cost of WS-only** (§4.6): whether the OS-capable PE logic actually gets dead-code-eliminated once the dataflow control signal is a compile-time constant — this project stopped at RTL/Verilator, before any synthesis pass.
- **`exp` FLOP-counting convention** (Phase 1a, §1.1): whether to count a hardware `exp` as 1 FLOP or weight it higher was flagged as an explicit, unresolved convention choice if softmax's own cost is ever revisited quantitatively (currently justified as negligible by the ~128× ratio argument, which doesn't depend on the convention).
- **GQA's real throughput payoff in a memory-bound regime** (Phase 1a Key Takeaway #7): prefill (compute-bound, fused) gets zero throughput benefit from GQA's bytes reduction — decode (Phase 2, memory-bound by nature) is where the same technique should show a genuinely different kind of payoff. Direct, pre-registered material for the Phase 1-vs-Phase 2 comparison.
- **Accumulator capacity above the 256 KB default** (Phase 1b, §2.3): logged as a real free parameter (128 KB–512 KB, grounded in Gemmini's published "BigSP" config) — never actually swept in Phase 1c/1d given the scope decisions made along the way.

---

## Appendix A — Farmshare / Toolchain Practical Notes

Operational lessons from actually running this project, worth having on hand before Phase 2 repeats the same infrastructure:

- **Environment split**: Timeloop/Accelergy (Docker-based) runs fine locally on the M3 Max — no need for Farmshare there. Chipyard/Gemmini/Verilator needs Farmshare (Linux x86_64 toolchain target).
- **Farmshare access requires interactive login (Duo 2FA)** — cannot be driven by a non-interactive SSH session; commands must be run by hand and pasted back for interpretation.
- **Always `source ~/chipyard/env.sh` before running `make`**, especially in a fresh `tmux` pane — a new pane does not inherit a prior shell's sourced environment. Running `make` before sourcing can bake a broken `$RISCV`-derived path into generated Makefiles, and merely fixing the env afterward does not force regeneration — requires a clean `rm -rf` of that config's `generated-src`/`simulator-*` directories and a fresh build.
- **Use `tmux` for anything long-running** (a first build of a new Gemmini config, or any Verilator run past a few seconds) — Chipyard elaboration/compile and longer sims can run for tens of minutes to hours, and a dropped SSH connection kills an un-`tmux`'d job outright.
- **Long-running Verilator binaries need `TIMEOUT_CYCLES` raised explicitly** (e.g. `TIMEOUT_CYCLES=100000000`) — the harness has a default max-cycle safety limit that silently truncates a simulation before natural completion otherwise.
- **The local `~/dev/chipyard` clone has the `gemmini` submodule uninitialized** — it's a stale/incomplete reference checkout, not a working copy. Editing Gemmini config files needs to happen directly on Farmshare (`nano`/`vim` over SSH); a git-based sync workflow between local and Farmshare was considered and rejected as more ceremony than a handful of small config edits warranted (editing inside a submodule means committing within the submodule's own repo *and* updating the superproject's pointer commit — real overhead for this scale of change).
- **Real hardware precedent for config choices should come from the actual target generator's own documented range, not an unrelated real chip** — Gemmini's own `Configs.scala` (`largeChipConfig` at 32×32) was the right basis for scaling down the array size; TPU-scale SRAM or TPU v1's accumulator were tempting but wrong comparison bases, twice.
