# Phase 1: Prefill Attention — Hand Analysis Notes

## Workload Shape (Llama 3-70B, prefill)

Source: *How to Scale Your Model*, Part 8 ("Serving LLaMA 3-70B on TPUs")

- batch = 32
- seq_len = 8192
- n_heads = 64
- n_kv_heads = 8 (GQA)
- d_head (d_qkv) = 128
- d_model = n_heads × d_head = 64 × 128 = 8192 (to verify against source)
- precision = int8 (params + KV cache)

**Approach:** derive naive MHA first as a baseline/control, then GQA, to isolate
the effect of KV-head sharing on arithmetic intensity rather than conflating
it with "attention in general."

---

## Pass 1: Naive MHA (n_kv_heads = n_heads = 64, control case)

### FLOPs — QK^T

Per (batch, head): Q(seq_len, d_head) × K^T(d_head, seq_len) → (seq_len, seq_len)
GEMM shape: M = seq_len, N = seq_len, K(contraction) = d_head

FLOPs = 2 × batch × n_heads × seq_len × seq_len × d_head
      = 2 × 32 × 64 × 8192 × 8192 × 128

**Status: derived, confirmed correct.**

### FLOPs — Softmax

Elementwise/reduction ops (max, subtract, exp, sum, divide) over the
(seq_len × seq_len) score matrix, per (batch, head).

FLOPs ≈ O(batch × n_heads × seq_len²) — small constant factor, vs. QK^T's
O(batch × n_heads × seq_len² × d_head).

Ratio QK^T : softmax ≈ d_head = 128×.

**Decision: softmax FLOPs treated as negligible relative to the matmuls**,
justified by the ~128× gap above (not just asserted — checked).

Open item: convention for counting `exp` (1 op vs. weighted higher) — state
explicitly in final writeup if softmax bytes/cost is revisited.

### FLOPs — Attention × V

Per (batch, head): weights(seq_len, seq_len) × V(seq_len, d_head) → (seq_len, d_head)
GEMM shape: M = seq_len, N = d_head, K(contraction) = **seq_len** (differs from QK^T,
where the contraction dim was d_head — numerically coincidental that the final
FLOP count matches QK^T for this particular shape, since seq_len appears
symmetrically either way).

FLOPs = 2 × batch × n_heads × seq_len × seq_len × d_head
      = 2 × 32 × 64 × 8192 × 8192 × 128

**Status: derived, confirmed correct (value matches QK^T, but via different
M/N/K assignment — not a shortcut).**

### Total FLOPs (MHA)

Softmax negligible (established above), so:

Total FLOPs = QK^T FLOPs + ·V FLOPs
            = 2 × (2 × 32 × 64 × 8192 × 8192 × 128)
            = 2^46
            ≈ 7.037 × 10^13 FLOPs

### Bytes Moved — Q/K/V/Output (compulsory traffic, perfect reuse)

Each of Q, K, V, and the final output is (batch, n_heads, seq_len, d_head),
read/written exactly once from/to HBM — this is a **lower bound**: it assumes
perfect on-chip reuse (no re-fetching due to tiling/scratchpad limits). Stated
explicitly as an assumption — real mappings may re-read data if scratchpad
can't hold what's needed, which is a predicted source of divergence from
Timeloop/Gemmini.

Elements per tensor = batch × n_heads × seq_len × d_head
                     = 32 × 64 × 8192 × 128 = 2^31 = 2,147,483,648

At int8 (1 byte/element): 2^31 bytes = 2 GiB per tensor.

Q read + K read + V read + output write = 4 × 2^31 = 2^33 bytes = **8 GiB**

### Bytes Moved — Softmax / P-matrix traffic (fused vs. unfused)

**Fusion assumption (both bounds computed, not just one — per methodology
discussion):**

- **Fused/tiled**: QK^T → softmax → ·V happen without the (seq_len × seq_len)
  score/probability matrix P ever touching HBM. Additional P-traffic = **0**.
- **Unfused**: P round-trips to HBM four times — QK^T write, softmax read,
  softmax write, ·V read.

**Precision decision for P (unfused case):** on-chip compute for softmax
(max, subtract, exp, sum, divide) is done at higher precision (fp16/32) for
numerical stability — this affects on-chip ALU/PE work only, **not**
bytes-moved. For every HBM write, P is **requantized back to int8** before
the write (and dequantized back to fp16/32 on the subsequent read). Rationale:
preserves the whole point of choosing int8 for memory-bandwidth savings;
tradeoff is added dequant/compute overhead and repeated-quantization accuracy
loss per round trip, judged worth it for the bandwidth savings on this
workload. (Real Rivos-relevant tradeoff — revisit explicitly in Phase 3.)

Given int8 throughout, all four unfused P round-trip terms are equal:

Elements per P round trip = batch × n_heads × seq_len × seq_len
                           = 32 × 64 × 8192 × 8192 = 2^37 = 137,438,953,472

At int8: 2^37 bytes = 128 GiB per round trip.

Unfused P-traffic = 4 × 2^37 = 2^39 bytes = **512 GiB**

### Total Bytes Moved (MHA)

- **Fused bound**: 2^33 bytes = **8 GiB**
- **Unfused bound**: 2^33 + 2^39 = 558,345,748,480 bytes ≈ **520 GiB**

(Q/K/V/output traffic is the same in both cases — 8 GiB — the entire gap
between the two bounds is the P-matrix round-trip traffic, ~512 GiB, i.e.
fusion is the dominant lever on bytes-moved for this shape, not Q/K/V access.)

### Arithmetic Intensity (MHA)

AI = Total FLOPs / Total Bytes

- **Fused bound**: 2^46 / 2^33 = **8192 FLOPs/byte**
- **Unfused bound**: 2^46 / 558,345,748,480 ≈ **126 FLOPs/byte**

Note: fused AI = unfused AI × 65, exactly mirroring the 65× bytes gap
identified earlier (2^33(1 + 2^6) factorization) — not a coincidence, a
direct structural consequence.

### Ridge Point

Source: *How to Scale Your Model*, TPU serving chapter — chapter uses **TPU
v5e** for the LLaMA 3-70B serving example, so ridge point is derived from the
same chip for internal consistency with the workload source.

TPU v5e, int8: peak FLOPs/s = 3.94e14, HBM BW = 8.2e11 bytes/s

C (ridge point) = peak FLOPs/s / HBM BW = 3.94e14 / 8.2e11 ≈ **480.5 FLOPs/byte**

(Sanity check against Pope's "~300" rule of thumb: same order of magnitude,
not identical — consistent with 300 being a rough heuristic for a different
chip generation/precision, not a universal constant. Confirmed rather than
assumed.)

### Prediction (Phase 1a conclusion)

- Fused AI (8192) >> C (480.5) → **decisively compute-bound**
- Unfused AI (126) << C (480.5) → **decisively memory-bound**

This workload is **not intrinsically compute-bound or memory-bound** — which
regime it falls into is determined almost entirely by one implementation
decision (does the softmax/P-matrix intermediate ever touch HBM), not by the
workload shape itself. Both bounds sit decisively on their respective side of
the ridge point (not a close call in either direction), so this isn't
sensitive to modest errors in the ridge-point assumption either.

**Open prediction to carry into Phase 1c**: expect Timeloop's mapper to find
something close to the fused bound if scratchpad is large enough to hold a
usable tile of the score matrix; if not, expect a mapping that lands
somewhere between the two bounds (partial spilling), which would itself be
informative about the real constraint. This is the first concrete
hand-analysis-vs-tool comparison point for Phase 1c/1d gap-hunting.

### Open / Not Yet Done (MHA pass)

- [x] Total FLOPs (sum QK^T + ·V, softmax negligible)
- [x] Bytes moved: Q, K, V, output read/write
- [x] Bytes moved: softmax/P intermediate — fused and unfused bounds, with
      explicit precision (requantization) assumption stated
- [x] Arithmetic intensity (FLOPs / bytes) — fused and unfused
- [x] Ridge point for assumed accelerator (TPU v5e, int8, from same source as
      workload shape — 480.5 FLOPs/byte)
- [x] Prediction: compute-bound or memory-bound, and why — **fused case
      decisively compute-bound, unfused case decisively memory-bound; regime
      is determined by fusion decision, not workload shape alone**

**Phase 1a: complete (MHA pass).**

---

## Pass 2: GQA (n_kv_heads = 8)

### FLOPs — QK^T, ·V, softmax

Confirmed (checked independently for both matmuls, not assumed by analogy)
that QK^T and ·V FLOPs are **identical to MHA**: both FLOP formulas are driven
by execution count (bs × num_q_heads), not by how many distinct K/V tensors
back the computation — KV-head sharing changes *which data* gets reused, not
*how many times* the matmul runs. Softmax stays negligible for the same
ratio-check reason as MHA.

**Total FLOPs, GQA = Total FLOPs, MHA = 2^46, exactly.**

### Bytes Moved — Q/K/V/output (compulsory, perfect on-chip reuse)

bs = 32, num_q_heads = 64, num_k_heads = 8, seq_len = 8192, d_head = 128.

| term | formula | elements | bytes (int8) |
|---|---|---|---|
| Q | bs·num_q_heads·seq_len·d_head | 2^31 | 2 GiB (unchanged from MHA) |
| K | bs·num_k_heads·seq_len·d_head | 2^28 | 256 MiB (down 8× from MHA) |
| V | bs·num_k_heads·seq_len·d_head | 2^28 | 256 MiB (down 8× from MHA) |
| output | bs·num_q_heads·seq_len·d_head | 2^31 | 2 GiB (unchanged from MHA) |
| **total** | | | **4.5 GiB** (MHA: 8 GiB) |

Q and output keep `num_q_heads` dependence (genuinely per-query-head
tensors); K and V shift to `num_k_heads` dependence (shared across each group
of 8 query heads) — under the same "compulsory traffic, perfect on-chip
reuse, lower bound" assumption stated in the MHA pass.

(Numeric plug-in for this table done by the assistant at the user's explicit
request — all formulas, the FLOPs argument above, and the interpretation
below are the user's own derivation.)

**Open question, deferred to Phase 1b (not resolved here):** the MHA
"perfect reuse" assumption only required one K-head's data to stay resident
on-chip long enough for *one* Q-head's QK^T. For GQA to actually hit the
`num_k_heads`-only fetch count, what has to be true about what's resident in
scratchpad simultaneously — does serving a whole group of 8 query heads off
one on-chip-resident K/V head require more simultaneous state than the MHA
case, and does that change how big the Phase 1b scratchpad hypothesis needs
to be for GQA vs. MHA?

### Bytes Moved — Softmax / P-matrix traffic (fused vs. unfused)

Confirmed byte-for-byte identical to MHA: P is inherently a per-query-head
object (it's QK^T for a specific query head), so its shape and all four
round-trip terms carry `num_q_heads`, unaffected by KV-head sharing. Fused
case collapses to 0, same mechanism as MHA.

Unfused P-traffic = **512 GiB** (unchanged from MHA).

### Total Bytes Moved (GQA)

- **Fused bound**: 4.5 GiB (compulsory only)
- **Unfused bound**: 4.5 GiB + 512 GiB ≈ **516.5 GiB** (554,587,652,096 bytes
  exactly)

### Arithmetic Intensity (GQA)

- **Fused bound**: 2^46 / 4.5 GiB ≈ **14,564 FLOPs/byte**
- **Unfused bound**: 2^46 / 554,587,652,096 ≈ **126.9 FLOPs/byte**

### Comparison to MHA / Analysis

- **K/V bytes**: exact 8× reduction (64/8 = 8), confirming the pre-registered
  prediction exactly, not just "roughly."
- **Total compulsory bytes only dropped 1.78×** (8 GiB → 4.5 GiB), not 8×,
  despite K/V individually shrinking exactly 8×. Reason: Q and output are
  structurally invariant to `num_k_heads` and were already half (4 of 8 GiB)
  of the MHA total — that fixes a floor on the total no matter how much K/V
  shrink. **General lesson**: an N× reduction in a subterm only produces an
  N× reduction in the total when that subterm dominates the total.
- **Unfused AI barely moved** (126 → 126.9) despite the 8× K/V win, because
  P-traffic dominates the unfused total by ~100× and is completely untouched
  by KV-head sharing — the compulsory-bytes savings are a rounding error
  against P-traffic in the unfused case. **Amdahl's Law framing**: GQA's 8×
  speedup only applies to the K/V *fraction* of total bytes-moved. Unfused,
  that fraction is ~1% of the total (P-traffic dominates), so the ceiling on
  overall improvement is barely moved regardless of how good the K/V-side
  technique is. Once fusion removes P-traffic, K/V's fraction of the
  (much smaller) remaining total jumps toward ~100%, and the same 8×
  technique now shows up almost in full in the total. Same technique, same
  8× local speedup, opposite payoff — purely a function of what fraction of
  the current bottleneck it touches, not of the technique itself.
- **Fused AI increased by an exact 16/9× ratio** (8192 → 14,564), mirroring
  the 16/9× bytes-reduction ratio (8 GiB / 4.5 GiB) — same "exact structural
  ratio, not coincidence" pattern as the 65× fused/unfused ratio noted in the
  MHA pass.

### Open / Not Yet Done (GQA pass)

- [x] FLOPs (confirmed identical to MHA)
- [x] Bytes moved: Q/K/V/output compulsory bound
- [x] Bytes moved: P-traffic, fused and unfused (identical to MHA)
- [x] Arithmetic intensity, fused and unfused
- [ ] Ridge-point comparison / explicit compute-vs-memory-bound conclusion for
      GQA (not yet stated)
- [ ] On-chip reuse requirement for GQA scratchpad sizing (flagged above,
      deferred to Phase 1b)

---

## Key Takeaways (Phase 1a, MHA + GQA) — for final writeup

1. **Regime is a design decision, not a workload property.** Whether this
   workload is compute- or memory-bound is governed almost entirely by
   whether the P-matrix touches HBM (fusion), not by seq_len/batch/head
   count. Fused → decisively compute-bound (AI 8192 vs. ridge ~480.5).
   Unfused → decisively memory-bound (AI 126 vs. ridge ~480.5). Same
   workload shape, opposite conclusion.

2. **FLOPs are blind to KV-head organization; bytes are not.** GQA leaves
   QK^T/·V FLOPs completely unchanged — compute cost is governed purely by
   query-head count. GQA is a pure memory-side (bytes-moved) lever, with zero
   compute-side effect. Clean separation of "compute lever" (shape/algorithm)
   from "memory lever" (data organization/fusion).

3. **GQA's payoff is regime-dependent (Amdahl's Law).** The "8× KV
   reduction" is real and exact, but whether it moves total AI depends on
   what fraction of total bytes-moved K/V represents. Unfused: K/V is ~1% of
   the total (P-traffic dominates) → 8× local win, ~0% total win. Fused: K/V
   was co-equal with Q/output → 8× local win becomes a real 16/9× total AI
   win. Same technique, same local speedup, opposite payoff — purely a
   function of what fraction of the current bottleneck it touches.

4. **Fix the dominant bottleneck before chasing secondary optimizations.**
   Fusion is the dominant lever for this workload (65× AI swing) — GQA is
   second-order *unless* fusion is already solved, after which GQA becomes
   the dominant remaining lever. That ordering (fusion first, then
   KV-sharing) is a design-priority conclusion, not just an arithmetic
   curiosity, and generalizes to any later stack of optimizations (e.g.
   Phase 3 precision changes on top of Phase 1/2 dataflow decisions): always
   ask what fraction of the *current* bottleneck a technique actually
   touches before crediting it with its in-isolation speedup.

5. **Methodological habits that paid off:** computing both fused and
   unfused bounds instead of picking one (surfaced the fusion/GQA
   interaction above, which a single-path analysis would have hidden); and
   deriving the ridge point from the workload's own source chip (TPU v5e)
   instead of trusting a remembered heuristic, then cross-checking against
   the heuristic anyway (agreed in order of magnitude, not exactly —
   informative in itself about heuristic precision/chip-generation
   specificity).

6. **Open thread into Phase 1b:** every bytes-moved number here assumes
   "perfect on-chip reuse," and that assumption is more demanding for GQA
   than MHA — GQA's win requires one K/V head to stay resident across 8
   query heads' worth of work, not just 1. Whether that's a reasonable ask of
   the scratchpad is an open, unresolved question that feeds directly into
   the Phase 1b sizing hypothesis.

7. **GQA's benefit is regime-dependent in a second, deeper way than the
   fused/unfused Amdahl's Law point above.** Roofline time is
   `max(FLOPs/peak_compute, bytes/peak_bandwidth)`. In **fused prefill**
   you're decisively compute-bound, so the FLOPs term sets execution time —
   and since GQA leaves FLOPs completely unchanged, GQA's bytes-moved
   reduction **does not reduce fused-prefill execution time at all**, even
   though it improves the AI number. Its real payoff there is elsewhere:
   lower scratchpad/on-chip pressure (directly experienced during the
   Phase 1b sizing derivation — GQA's KV-sharing is what made the group's
   reuse fit in a realistic accumulator/scratchpad budget) and smaller
   KV-cache footprint/energy. In **decode** (Phase 2), which is
   memory-bound by the workload's own nature, the same bytes-moved
   reduction should translate directly into a throughput win — meaning
   GQA's justification is genuinely different in kind, not just degree,
   between the two regimes. Direct material for the Phase 2 "why does ideal
   hardware differ between prefill and decode" comparison.

---

## Phase 1b: Hardware Hypothesis (PE Array, Dataflow, Scratchpad Sizing)

Goal (per spec): a defensible hypothesis to test against Timeloop in 1c —
not required to be correct yet.

### 1. PE array shape

- Initial instinct: size the array to match the full GEMM dims directly
  (e.g. 8192×8192 for QK^T). **Rejected** after sanity-checking against real
  hardware: TPU MXU 256×256, Trainium 128×128, Nvidia tensor core ~16×16 —
  all orders of magnitude smaller than `seq_len`. Array size is a hardware
  design choice, independent of workload dimensions; `seq_len` must instead
  be tiled to pass through a much smaller array.
- `d_head` = 128 identified as a strong candidate for one array axis: it's a
  fixed dimension native to Q/K/V's own tensor shape (not workload-scale-
  dependent like `seq_len`), and appears in *both* matmuls — as the
  contraction dim (K) in QK^T, and as the output dim (N) in ·V.
- **Resolved via first-principles systolic-array framework:** a systolic
  array can only make 2 of a GEMM's (M,N,K) dims spatial at once; the
  dataflow name (weight-/output-/input-stationary) *is* the choice of which
  pair, and the 3rd dim streams/accumulates over time regardless of its
  size. Worked out, under the K/V-stationary choice from §2 (a
  weight-stationary framing): QK^T → spatial = (`d_head`, `seq_len_k`-tile),
  temporal = `seq_len_q`; ·V → spatial = (`seq_len_k`-tile, `d_head`),
  temporal = `seq_len_q`. **Both matmuls need the identical pair of axis
  sizes** (`d_head`, k-tile), just potentially transposed — a direct
  consequence of `d_head` being the one dimension shared by every operand
  (Q, K, V) under weight-stationary, not a coincidence. This resolves the
  d_head-role question: no cross-matmul utilization conflict, **provided**
  the array/feed logic can route either GEMM dimension onto either physical
  axis between the two matmul phases (stated assumption, not yet verified
  against real Gemmini capability — carry into Phase 1d).
- **Second array axis, decided:** no real area/power budget exists to solve
  this exactly, and array width is itself one of Timeloop's 1c sweep
  dimensions — so rather than force a single number, **locked in 128×128 as
  the primary hypothesis** (matches Trainium precedent exactly, not just an
  aesthetic "arrays are square" guess) **with 128×256 carried as an explicit
  alternate** to test (roughly halves the passes needed through the
  scratchpad-resident k-tile at ~2× array area/power cost; no new
  utilization penalty found between QK^T and ·V at either size).

### 2. Dataflow

- Defined "stationary" concretely: which of a GEMM's M/N/K dims sit
  *spatially* on the array (loaded once, fixed across many cycles) vs.
  stream through *temporally* — this is the literal content of
  weight-/output-/input-stationary, not just a label.
- **MHA**: no cross-head reuse for either Q or K (each head has unique data)
  → dataflow choice doesn't matter for MHA from a reuse standpoint.
- **GQA**: K (and, by identical logic, V) are reused across the group of
  `num_q_heads / num_k_heads` = 8 query heads sharing one KV head.
  **Concluded: K/V stationary, Q streamed** — directly the same reuse factor
  (8×) that drove the Phase 1a GQA byte savings, not a separately-derived
  fact.
- Self-corrected an initial inversion ("streaming K/V saves memory
  traffic") once explicitly tied back to the Phase 1a mechanism: reuse
  *without* re-fetching = stationary, not streaming. Also reasoned that
  Q being the larger tensor argues *against* making it stationary (no reuse
  benefit, more scratchpad cost for nothing) — same conclusion via a second
  angle.

### 3. Scratchpad & accumulator sizing (resolved)

- Distinguished array-level "stationary" (fine-grained, holds for one
  tile-pass) from scratchpad-level residency (must span the *entire*
  8-head group for GQA's compulsory-byte claim from Phase 1a to actually
  hold) — different levels of the memory hierarchy; conflating them was an
  early mistake, caught and corrected.
- First estimate: one full (batch, kv-head) K tile = `seq_len_k × d_head` =
  2^20 = **1 MiB** at int8 — checked against realistic on-chip SRAM sizes,
  not implausible.
- Reconnected to the Phase 1a fused/unfused split: a **full** P tile
  (`seq_len_q × seq_len_k`) = 2^26 = **64 MiB** per head — far too large for
  any realistic scratchpad. Conclusion: achieving "fused" (P never touches
  HBM, the Phase 1a compute-bound case) forces tiling of `seq_len_k`
  (likely `seq_len_q` too) *regardless of GQA* — the two tiling
  requirements (P-size-driven, and GQA-reuse-driven) compose rather than
  conflict.
- Revised resident-K/V estimate once tiled: `tile_size × d_head × 2` (K and
  V for one chunk) — much smaller than the full 1 MiB.
- Identified a real wrinkle: preserving GQA's KV-chunk reuse under this
  tiling requires the loop order to hold a K/V chunk fixed while sweeping
  across all 8 heads in the group (not finishing one head's full sweep
  before the next, which would defeat the reuse) — this requires per-head
  online-softmax correction/tracking state carried across the KV-chunk
  sweep. Independently reconstructed the core idea behind Flash Attention's
  online-softmax mechanism from first principles (memory-traffic
  constraints), before being told the name.
- **Real-hardware SRAM sanity check (verified via web search, not assumed):**
  briefly considered Google's newly-announced (Apr 2026) TPU 8t/8i (128 MiB /
  384 MiB on-chip SRAM respectively) — rejected as the sizing target, both
  because it breaks internal consistency with the TPU v5e ridge point from
  Phase 1a, and because it's wildly larger than anything realistically
  RTL-simulatable in Gemmini for Phase 1d. Also checked TPU v5e's own VMEM
  directly (128 MiB, from the same *How to Scale Your Model* source as the
  workload) to correct an initial misremembered "32 MB." **Decision: anchor
  scratchpad sizing to real Gemmini defaults**, not TPU-scale SRAM — verified
  via Gemmini's GitHub/paper: base config = **256 KB scratchpad** (up to
  ~1 MiB across banks in some configs) + **256 KB accumulator**, as two
  *separate* physical memories, not one combined pool.
- Resolved a conceptual question along the way: scratchpad sizing isn't
  "whatever's left over after the PE array" — there's a principled minimum
  (driven by the reuse pattern already derived) past which more SRAM buys no
  further bytes-moved reduction, and Phase 1c's Timeloop sweep explicitly
  searches memory sizing too, so an unfalsifiable "as big as possible"
  hypothesis would give nothing to compare against that sweep.
- **Memory-pool placement:** the K/V chunk (int8, stationary operand data)
  belongs in **scratchpad**; the online-softmax tracking state (higher
  precision, per Phase 1a's own fp16/32 softmax-precision decision) belongs
  in **accumulator** — matching accumulator's architectural purpose of
  holding higher-precision partial sums during accumulation.
- **Per-head tracking state, resolved:** not just one number per head —
  running max (`tile_q` elements) + running sum (`tile_q` elements) +
  partial output accumulator (`tile_q × d_head` elements, the dominant term
  since it scales with `d_head`=128). At fp32 (4 bytes/elem): per head =
  `tile_q · (2 + d_head) · 4` = `520 · tile_q` bytes.
- **Group-size clarification:** the "8" scaling the per-head state is
  `num_q_heads / num_k_heads` (= 64/8 = 8), the number of query heads
  sharing one KV head — **not** `num_k_heads` directly. It's a coincidence
  of this particular shape (64 = 8²) that the group size and `num_k_heads`
  are numerically identical; the formula must reference the group size, not
  `num_k_heads`, for the reasoning to generalize to other shapes.
- **Solved for `tile_q`:** `8 heads × 520 · tile_q bytes ≤ 256 KiB (262,144
  bytes)` → `tile_q ≤ 262,144 / 4,160 ≈ 63.0` → **max tile_q = 63 elements**
  (4160×63 = 262,080 B, fits; ×64 = 266,240 B, doesn't). Hardware-friendly
  power-of-2 choice: **tile_q = 32**.
- **Resolves the earlier open question:** yes, `seq_len_q` must be tiled as
  aggressively as `seq_len_k` — the accumulator's per-head tracking cost,
  multiplied across the group of 8, forces a ~256× reduction from 8192 down
  to a ~32-element query tile, a far tighter constraint than the K/V-chunk
  sizing alone would have suggested.
- **Full scratchpad inventory (not just K/V):** enumerated every simultaneous
  claimant on the scratchpad budget — K tile ×2 and V tile ×2
  (double-buffered, to overlap next-chunk HBM prefetch with current-chunk
  compute), the P/S intermediate tile, the Q staging tile (no direct
  HBM→array datapath — everything stages through scratchpad first), and the
  output tile before HBM writeback.
- **Solved for `tile_k`:** with `tile_q`=32 and `d_head`=128 fixed — K×2 + V×2
  = `512·tile_k` bytes, P (`tile_q × tile_k`) = `32·tile_k` bytes, Q + output
  (`tile_q × d_head` each) = `4,096 + 4,096` = `8,192` bytes (fixed). Total =
  `544·tile_k + 8,192` bytes ≤ 1 MiB (1,048,576 B) → `tile_k ≤ 1,912.5` →
  practical **tile_k = 1024** (~565 KB used, leaving headroom in the 1 MiB
  scratchpad budget).
- **Second layer of tiling, caught before finalizing:** `tile_k`=1024 is
  itself far larger than any realistic array physical size (~128–256, per
  the earlier hardware check) — so `tile_k` is a *scratchpad-residency*
  number, not the array's spatial dimension. The array sweeps through the
  1024-element scratchpad-resident chunk in ~1024/128 = 8 sub-passes,
  mirroring the original `seq_len`→array relationship one level down. This
  directly fed the final PE-array-shape resolution in §1.

---

## Phase 1b: Final Hypothesis (for testing against Phase 1c)

- **PE array**: **128×128** (`d_head` × k-sub-tile) as the primary
  hypothesis — matches Trainium precedent exactly. **128×256 carried as an
  explicit alternate**, to test whether the ~2× pass-count reduction through
  the scratchpad-resident k-tile is worth ~2× the array area/power; no
  utilization conflict expected between QK^T and ·V at either size.
- **Dataflow**: weight-stationary, with **K and V as the stationary
  operand** in their respective matmuls (QK^T and ·V) — driven by the 8×
  reuse factor (`num_q_heads/num_k_heads`) across the group of query heads
  sharing one KV head; Q streamed (no cross-head reuse to exploit). The same
  physical array serves both matmuls because `d_head` anchors the spatial
  pair in both, just potentially transposed.
- **Scratchpad** (Gemmini default, ≤1 MiB): holds double-buffered K/V chunks
  (`tile_k`=1024, re-solved below) + a *fixed-size* P tile (`tile_q×128`,
  not `tile_q×tile_k` — see granularity note) + Q tile + output tile.
- **Accumulator** (Gemmini default, ≤256 KiB): holds (a) per-head
  online-softmax tracking state (running max + running sum + partial output
  accumulator, fp32) for the group of 8 query heads sharing one KV head —
  `tile_q`=32, ≈130 KB across 8 heads — **plus (b) a transient raw S/P
  block**, sized `tile_q × 128` (fine-grained — see below), ≈16 KB, not
  multiplied by 8 heads. Found late: initially missing from the `tile_q`
  solve entirely; the *coarse* version of this term (`tile_q × tile_k` =
  ~128 KB) was checked and does **not** fit alongside the 8-head running
  state (133,120 + 131,072 = 264,192 B > 262,144 B budget, over by 2 KB) —
  this is what forced the granularity decision below, not just a stylistic
  preference.
- **Pipeline note / softmax granularity — resolved fine-grained:** raw S/P
  (pre-softmax, higher precision) lives in the accumulator; only the
  *post*-softmax, requantized-to-int8 P lives in scratchpad as the ·V
  operand. Softmax runs **per 128-wide array sub-pass** (fine-grained), not
  once per full `tile_k` chunk (coarse) — the coarse version was the
  original hypothesis but was found to overflow the accumulator budget by
  2 KB once the raw-S term was properly counted; fine-grained fits with
  large margin (16 KB vs. 128 KB) and doesn't require inventing a new
  precision cut, at the cost of 8× more control/softmax-touch instructions
  per chunk (accepted tradeoff). This also converges toward the standard
  Flash-Attention structure (one K/V block through the full QK^T→softmax→·V
  pipeline before advancing), rather than an ad-hoc alternative. Softmax's
  own ops (max, subtract, exp, sum, divide, rescale) are elementwise/
  reduction, not matmul — run on a separate vector/scalar unit
  reading/writing the accumulator, array idle throughout; whether Gemmini
  has a native unit for this or needs the host Rocket/BOOM core is an open
  Phase 1d question.
- **`tile_k` re-solved under fine-grained P** (now a fixed `4,096`-byte
  term instead of scaling with `tile_k`): `512·tile_k + 12,288 ≤ 1,048,576`
  → `tile_k ≤ 2,024` — nearly double the old ceiling (1,912). But
  `seq_len_k`=8192=2¹³, so only power-of-2 `tile_k` values divide it evenly
  (avoiding a ragged final chunk); the next power of 2 above 1024 is 2048,
  which exceeds the new 2,024 ceiling. **`tile_k`=1024 stands unchanged** —
  not a compromise, but still the correct answer given the divisibility
  constraint, now for a sharper reason than the original "power-of-2 feels
  hardware-friendly" heuristic.
- **Stated assumptions carried into 1c/1d:** (a) array/feed logic can route
  either GEMM dimension onto either physical axis between the QK^T and ·V
  phases; (b) fp32 for online-softmax tracking state (revisit at fp16 in
  Phase 3); (c) double-buffering assumed for K/V but not for Q/output; (d)
  **fine-grained** softmax granularity (per 128-wide array sub-pass, not per
  `tile_k` chunk); (e) softmax executes on an as-yet-unspecified
  vector/scalar unit, not the systolic array.
- **Predictions to check against Timeloop:** does the mapper's near-optimal
  config land closer to 128×128 or 128×256? Does it find `tile_k`/`tile_q`
  close to the hand-derived 1024/32, or reveal a consideration missed here?
  This is the first concrete hand-vs-tool comparison point for Phase 1b,
  mirroring the fused/unfused prediction from Phase 1a.

### Major open finding: K/V reuse vs. accumulator capacity (loop-order tension)

Discovered by walking the full loop nest explicitly with all levels present
(batch, KV-group, Q-tile, K/V-chunk, head, array sub-pass), not just
reasoning about each piece in isolation:

- `tile_q`=32 was solved against the accumulator budget assuming **8 heads
  × 1 Q-tile** of simultaneous running state (`8 heads × 520·tile_q ≤
  256 KiB` — no factor for holding multiple Q-tiles at once). The
  accumulator physically cannot hold in-progress online-softmax state for
  more than one Q-tile at a time.
- This forces **Q-tile to be the outer loop relative to K/V-chunk**: you
  must finish one Q-tile (cycling through all 8 K/V-chunks) before starting
  the next, because the accumulator can't hold multiple Q-tiles' state
  concurrently to let chunk stay outer instead.
- **Consequence: this breaks the "K/V-chunk fetched once per group" GQA
  reuse assumption.** With Q-tile outer, each K/V-chunk (evicted from the
  ~1 MiB scratchpad to make room for the next chunk during one Q-tile's
  sweep) has to be **re-fetched from HBM for every one of the 256 Q-tiles**,
  not fetched once per (batch, kv-head) group as Phase 1a's compulsory-bytes
  lower bound assumed.
- **This is not a broken hypothesis — it's the exact category of gap Phase
  1a's own bytes-moved section predicted**: *"this is a lower bound... real
  mappings may re-read data if scratchpad can't hold what's needed, which is
  a predicted source of divergence from Timeloop/Gemmini."* Finding a
  concrete, hand-derivable instance of that predicted gap is the prediction
  paying off, not a failure. All FLOPs work, the roofline/AI methodology,
  ridge-point derivation, Amdahl's-Law insight, and dataflow/array-shape
  derivation are entirely unaffected — this only concerns whether *this
  specific* `tile_q`=32 config achieves full vs. partial K/V byte-savings.
- **Real, unexplored design lever**: `tile_q`=32 was chosen to *maximize*
  Q-tile size (minimizing softmax/accumulator-touch instruction count) —
  one extreme of a tradeoff. A *smaller* `tile_q` would free accumulator
  room to hold multiple Q-tiles' state simultaneously, directly reducing
  K/V re-fetch frequency, at the cost of more sub-passes/instructions
  overall. This is a genuine multi-dimensional tradeoff (tile size ×
  re-fetch frequency × instruction count) intentionally left for Timeloop's
  mapper to search rather than hand-optimized further, per the project's
  own "defensible hypothesis, not optimal" framing for Phase 1b.
- **Sharpened prediction for Phase 1c**: does Timeloop's mapper converge to
  something like `tile_q`=32 (accepting heavy K/V re-fetching), or does it
  find a smaller `tile_q` that trades instruction count for better reuse?
  This is now a specific, falsifiable comparison point, not just "see what
  the mapper finds."

---

## Key Takeaways (Phase 1b) — for final writeup

1. **Array sizing is governed by real hardware precedent and dataflow-driven
   reuse, not workload scale.** `seq_len` (8192) never appears directly in
   the array's shape at any level — it gets tiled away twice, first to a
   scratchpad-resident chunk, then again to an array-sized sub-tile — a
   pattern that repeated itself once you knew to look for it.
2. **Dataflow and array shape aren't separable questions.** "What shape
   should the array be" has no answer until "which operand is stationary"
   is fixed, since the dataflow choice determines which GEMM dimensions even
   compete for the array's two physical axes.
3. **`d_head` being shared across every operand (Q, K, V) is why one
   physical array serves both QK^T and ·V cleanly** under weight-stationary
   — a structural property of attention's shape, discovered by working out
   each matmul's spatial/temporal split independently and noticing they
   matched, not assumed in advance.
4. **Scratchpad and accumulator are separate physical resources with
   different natural occupants**, mirroring the Phase 1a precision
   decision directly onto physical memory placement: scratchpad holds int8
   stationary/streamed data (K, V, Q, P, output); accumulator holds the
   higher-precision partial-sum/tracking state.
5. **Tiling is a two-level phenomenon, easy to conflate.** Workload scale
   (`seq_len`) reduces to a scratchpad-resident chunk under one budget
   (scratchpad capacity), then that chunk reduces again to an array-sized
   sub-tile under a second, independent budget (real array size) — catching
   the conflation was itself one of the most useful moments in Phase 1b.
6. **When a tradeoff can't be solved exactly (no real area/power budget for
   square vs. 128×256), state a defensible primary hypothesis plus an
   explicit alternate** rather than forcing a single answer — especially
   when the tool you're about to use (Timeloop) sweeps exactly that
   dimension anyway.
7. **GQA's benefit is regime-dependent in two distinct ways**, discovered
   across Phase 1a and 1b respectively: Amdahl's-Law-style (fused vs.
   unfused bytes-moved) and roofline-position-style (fused prefill's
   compute-bound time isn't helped by GQA at all — its real payoff there is
   scratchpad pressure and KV-cache footprint, not throughput).
8. **"Stationary" is always scoped to a specific memory boundary, and this
   pattern recurs recursively.** Recognized independently three times:
   scratchpad-vs-accumulator, array-vs-scratchpad (`tile_k` sub-passes),
   and HBM-vs-scratchpad (K/V "stationary" means HBM isn't re-touched per
   head, not that the array's PE contents are frozen). Whenever something
   is called "resident," the next question is always *relative to which
   boundary*.
9. **Raw matmul output and post-processing output are different objects,
   even when they coincidentally match in size.** The pre-softmax S block
   (accumulator) and the post-softmax, requantized P (scratchpad) looked
   identical (`32×128` each) purely because the primary array hypothesis is
   square — checking against the 128×256 alternate is what proved they're
   independent terms, exposing a genuinely missing accumulator line item.
10. **A choice made for one reason can silently violate a different,
    unrelated constraint.** Coarse softmax granularity was chosen to
    minimize control instructions — but once the raw-S term was counted
    honestly, coarse didn't fit the accumulator budget at all (over by
    2 KB). The forced fix (fine-grained) also happened to converge toward
    how real Flash Attention actually works.
11. **Workload shape can make a "hardware-friendliness heuristic" into a
    hard requirement.** `seq_len_k`=8192=2¹³ means only power-of-2 tile
    sizes divide it evenly (avoiding a ragged final tile) — so `tile_k`
    staying at 1024 even after its ceiling nearly doubled wasn't a style
    choice, it was the only valid answer given the workload's own shape.
12. **Major finding: capturing GQA's cross-head reuse and having enough
    per-head Q-tile capacity are in direct tension under a small, fixed
    accumulator.** Sweeping 8 heads together against one resident K/V chunk
    (the mechanism behind GQA's HBM savings) requires holding 8 heads'
    state simultaneously, which leaves room for only one Q-tile at a time,
    which forces the *opposite* problem: re-fetching each K/V chunk ~256×
    instead of once. The mechanism that lets you exploit the reuse is what
    starves the capacity needed to avoid re-fetching. Not a sign GQA
    "doesn't work" (the computation is unaffected) — a sign this specific
    config likely doesn't realize the full theoretical 8× byte-savings.
    This is exactly the category of gap Phase 1a's own bytes-moved section
    predicted in advance ("real mappings may re-read data if scratchpad
    can't hold what's needed") — finding a concrete instance of it by hand
    is the prediction paying off, not a setback.
13. **When you hit a wall like #12, check whether it's fundamental or an
    artifact of one specific choice.** Here it's the latter: a smaller
    `tile_q` trades more control instructions for holding multiple Q-tiles
    simultaneously, directly reducing re-fetch frequency — a real,
    unexplored lever, deliberately left unresolved as a sharpened
    prediction for Timeloop rather than hand-optimized further.
14. **Cross-chip comparisons need to match constraint regime, not just be
    factually real.** TPU v1's 4 MB accumulator is verified-real, but
    citing it to justify a Gemmini config is the wrong comparison basis
    (custom datacenter ASIC vs. small RTL generator) — the same category of
    mistake as reaching for TPU 8t/8i's SRAM earlier. A true fact can still
    be an invalid argument if the underlying chips aren't comparable.
15. **The right resolution was checking the actual target tool's own
    documented flexibility, not reaching for an unrelated chip.** Gemmini's
    `acc_capacity` is a real, user-configurable generator parameter, and a
    real, published, *benchmarked* config (BigSP: 512 KB scratchpad + 512 KB
    accumulator) confirmed it's realistic — turning speculation into a
    grounded search-range endpoint for Phase 1c.
16. **Real data can be calibrating, not just confirming.** BigSP's measured
    speedup on Matmul-category workloads was small (~1-3%) versus Conv's
    (~10-11%) — a genuine caution against assuming "bigger accumulator"
    proportionally fixes the re-fetch tension from #12. Worth carrying into
    how Phase 1c's actual results get interpreted.
17. **How Phase 1b and 1c actually divide labor.** 1b's job was producing
    specific, justified predictions wherever hand-derivation could reach
    one (dataflow, `d_head` as an axis, `tile_k`=1024, `tile_q`=32) — and
    explicitly flagging, not forcing, the places it couldn't (array's
    second axis, accumulator capacity, the reuse/capacity tension). 1c
    isn't limited to picking from a hand-curated shortlist; it's a genuine
    search, with the hand-derived numbers serving as falsifiable comparison
    points, not constraints on what the tool is allowed to find.

---

## Open / Not Yet Done (Phase 1b)

- [x] PE array shape (128×128 primary, 128×256 alternate)
- [x] Dataflow (weight-stationary, K/V stationary, same array for both
      matmuls)
- [x] Scratchpad sizing (`tile_k`=1024)
- [x] Accumulator sizing (`tile_q`=32)
- [ ] Verify the array axis-routing/transpose assumption against actual
      Gemmini capability (Phase 1d)
- [ ] Test 128×128 vs. 128×256 against Timeloop's actual sweep (Phase 1c)
- [ ] **Softmax execution unit (Phase 1d):** softmax's elementwise/reduction
      ops (max, subtract, exp, sum, divide, online-rescale correction) don't
      fit the systolic array's MAC-only structure — they need a separate
      vector/scalar functional unit reading/writing the accumulator
      directly, with the array idle during this step. Check whether Gemmini
      has a native unit for this or whether it has to route through the
      host Rocket/BOOM core — the latter would be a real dispatch/data-
      movement overhead that hand analysis and possibly even Timeloop's
      cost model won't capture, and a prime gap-hunting target for Phase 1d.
      Connects to the "convention for counting exp" open item from Phase 1a.
- [ ] **K/V reuse vs. accumulator capacity (Phase 1c):** `tile_q`=32 only
      supports 1 Q-tile's worth of simultaneous accumulator state, forcing
      Q-tile outer to K/V-chunk in the loop nest and breaking the
      "fetched once per group" GQA reuse assumption from Phase 1a (each
      chunk re-fetched ~256× instead of 1×, under this specific config).
      Intentionally left unresolved by hand — check whether Timeloop's
      mapper finds a smaller `tile_q` that trades instruction count for
      better reuse, or converges to the same heavy-re-fetch tradeoff.
- [ ] **Accumulator capacity as a free parameter (Phase 1c):** 256 KB is
      Gemmini's *base/default* `acc_capacity`, not a hard ceiling. Considered
      using TPU v1's 4 MB accumulator as justification for a bigger number,
      rejected that comparison basis (custom datacenter ASIC vs. small RTL
      generator — same category of mistake as the earlier TPU 8t/8i
      comparison). **Confirmed instead via a real Gemmini paper figure**: a
      published, benchmarked "BigSP" config exists with 512 KB scratchpad +
      512 KB accumulator (+1 MB L2) — so 512 KB is a real, realistic
      Phase 1c/1d target, not just plausible. That same figure's measured
      Matmul-category speedup for BigSP was small (~1-3%) vs. Conv's
      ~10-11%, though — don't assume bigger accumulator proportionally
      fixes the re-fetch tension; treat as a real, sobering expectation to
      check against, not a solved problem. Still log accumulator capacity as
      an explicit free parameter for Timeloop's 1c sweep (128 KB base
      through 512 KB BigSP-confirmed, possibly higher), alongside array
      shape and `tile_k`/`tile_q`.

**Phase 1b: complete** (with two open findings carried into 1c: the
K/V-reuse/accumulator-capacity tension above, and accumulator capacity
itself as an unresolved free parameter).
Ready for Phase 1c (Timeloop sweep).

---

## Log

- Derived QK^T FLOPs (MHA) — confirmed via M/N/K decomposition.
- Decided softmax FLOPs negligible — confirmed via ratio check (~128×), not
  just assumed.
- Flagged fusion assumption as an open, consequential decision for bytes-moved.
- Derived ·V FLOPs (MHA) — confirmed via independent M/N/K decomposition
  (contraction dim = seq_len, not d_head).
- Derived Q/K/V/output bytes-moved (compulsory/perfect-reuse lower bound),
  int8, 2 GiB per tensor.
- Resolved fusion question: compute **both** fused and unfused bounds rather
  than picking one, since the gap itself is predicted, explainable divergence
  material for later phases.
- Resolved precision question for P-traffic: requantize to int8 before every
  HBM write (bandwidth-savings rationale over requant compute/accuracy cost);
  on-chip softmax compute stays higher precision but that's invisible to
  bytes-moved.
- Totaled FLOPs (2^46) and bytes-moved for both bounds (fused: 8 GiB; unfused:
  ~520 GiB, ~512 GiB of which is P-matrix round-trip traffic alone).
- Totaled FLOPs (2^46) and bytes-moved for both bounds (fused: 8 GiB; unfused:
  ~520 GiB, ~512 GiB of which is P-matrix round-trip traffic alone).
- Computed AI for both bounds: fused = 8192, unfused ≈ 126 (exact 65× ratio,
  matching the earlier bytes-gap factorization).
- Derived real ridge point from TPU v5e int8 specs (same chip as workload
  source) instead of a recalled heuristic: C ≈ 480.5. Cross-checked against
  Pope's "~300" rule of thumb — same order of magnitude, not identical,
  consistent with heuristic being chip/precision-specific.
- Closed out Phase 1a prediction: workload's compute/memory-bound regime is
  governed by the fusion decision (softmax/P on-chip or not), not the
  workload shape alone — both bounds land decisively on their side of the
  ridge, so conclusion is robust to modest ridge-point error.
- **Phase 1a (MHA) complete.** Next: GQA pass.
- Confirmed GQA FLOPs (QK^T, ·V, softmax) identical to MHA — checked
  independently for both matmuls (execution count is driven by
  `num_q_heads`, not by distinct K/V tensor count), not assumed by analogy.
- Derived GQA Q/K/V/output byte-term formulas: Q and output keep
  `num_q_heads` dependence, K and V shift to `num_k_heads` dependence, under
  the same MHA "compulsory, perfect on-chip reuse" assumption.
- Flagged open question, deferred to Phase 1b: what on-chip residency does
  GQA's `num_k_heads`-only fetch count actually require — does serving 8
  query heads off one resident K/V head demand more simultaneous scratchpad
  state than MHA's single-head case?
- Confirmed GQA P-traffic (fused/unfused) is byte-for-byte identical to
  MHA — P is a per-query-head object regardless of KV-head sharing.
- Numeric plug-in for GQA bytes/AI computed by the assistant at the user's
  explicit request; formulas and interpretation are the user's own.
- Confirmed pre-registered "~8× K/V byte reduction" prediction exactly
  (64/8 = 8×).
- Explained why total compulsory bytes only dropped 1.78× despite the exact
  8× K/V reduction: Q+output are invariant to `num_k_heads` and were already
  half the MHA total, capping the achievable total reduction. General lesson
  logged: a subterm's N× reduction only yields the total's N× reduction if
  that subterm dominates the total.
- Explained why unfused AI barely moved (126 → 126.9): P-traffic dominates
  the unfused total (~100× the compulsory bytes) and is entirely untouched by
  KV-head sharing.
- Computed GQA AI: fused ≈ 14,564 (exact 16/9× MHA's 8192, matching the 16/9×
  bytes-reduction ratio), unfused ≈ 126.9.
- **Phase 1a (MHA + GQA) FLOPs/bytes/AI derivation complete.** Ridge-point
  comparison / explicit compute-vs-memory-bound conclusion for GQA not yet
  stated.
- Reframed the GQA-payoff finding explicitly as an Amdahl's Law instance
  (user's framing): a technique's local speedup only shows up in the total
  in proportion to the fraction of the current bottleneck it touches —
  identified as a reusable check for evaluating any future stacked
  optimization (Phase 3 precision on top of Phase 1/2 dataflow, etc.).
- Added a "Key Takeaways" section consolidating Phase 1a's six real
  insights for the final writeup (regime-as-design-decision, FLOPs/bytes
  lever separation, Amdahl's-Law GQA payoff, fix-dominant-bottleneck-first
  priority ordering, methodological habits, open on-chip-reuse thread).
- **Phase 1a (MHA + GQA) fully logged and up to date.** Ready for Phase 1b
  (PE array shape, dataflow, scratchpad sizing hypothesis).
- Rejected sizing the PE array to full GEMM dims (8192×8192) after checking
  real hardware (TPU MXU 256×256, Trainium 128×128, Nvidia ~16×16) —
  confirmed array size is a hardware choice independent of workload scale.
- Identified `d_head`=128 as a candidate array axis (native to Q/K/V shape,
  appears in both matmuls); flagged its differing role (contraction dim in
  QK^T vs. output dim in ·V) and the second array axis as open.
- Defined "stationary" concretely (which GEMM dim sits spatially vs. streams
  temporally) and derived the dataflow conclusion: no reuse advantage either
  way for MHA; K/V stationary + Q streamed for GQA, driven by the same 8×
  reuse factor as the Phase 1a byte savings. Self-corrected an initial
  inversion of which operand streaming actually saves traffic.
- Verified TPU 8t/8i on-chip SRAM claim via web search (real, announced Apr
  2026 — 128 MiB / 384 MiB) — rejected as sizing target (breaks v5e ridge-
  point consistency; unrealistic for Gemmini RTL simulation in Phase 1d).
- Corrected a misremembered TPU v5e VMEM figure (32 MB) to the verified 128
  MiB, from the same workload-source book.
- Verified real Gemmini defaults via web search: 256 KB scratchpad (up to
  ~1 MiB across banks) + 256 KB accumulator, as separate physical memories —
  adopted as the realistic Phase 1b sizing target instead of TPU-scale SRAM.
- Resolved conceptual question: scratchpad sizing has a principled minimum
  (driven by the reuse pattern), not "whatever's left over after the PE
  array" — oversizing has no further bytes-moved benefit past that minimum,
  and Timeloop's 1c sweep over memory sizing needs a falsifiable hypothesis
  to compare against.
- Placed K/V chunk in scratchpad (int8, stationary data) and online-softmax
  tracking state in accumulator (higher precision, per Phase 1a's own
  softmax-precision decision) — two separate pools, not one combined budget.
- Derived per-head tracking-state composition (running max + running sum +
  partial-output accumulator, the last being the dominant, `d_head`-scaled
  term) and its byte formula (520·tile_q at fp32).
- Caught and corrected a scaling error: the group size is `num_q_heads /
  num_k_heads` (=8), not `num_k_heads` directly — numerically identical only
  by coincidence of this shape (64=8²).
- Solved for `tile_q` against the 256 KB accumulator budget: max ≈63
  elements exactly, tile_q=32 as the hardware-friendly power-of-2 choice —
  closing the earlier open question of whether `seq_len_q` needs tiling as
  aggressively as `seq_len_k` (yes, ~256× reduction).
- **Phase 1b scratchpad-sizing thread resolved and logged.** Remaining
  before 1b is "closed": settle the second PE array axis (likely = the tile
  size just derived) and the d_head-role question, then consolidate into one
  stated hypothesis before moving to Phase 1c (Timeloop sweep).
- Realized (unprompted, user's own catch) that GQA's fused-prefill bytes
  reduction doesn't move the roofline time bound at all, since compute-bound
  time is set by FLOPs alone and GQA leaves FLOPs unchanged — its real
  payoff there is scratchpad pressure and KV-cache footprint, not raw
  throughput; decode should be the regime where the throughput win actually
  shows up. Added as Key Takeaway #7, direct material for Phase 2 comparison.
- Solved the full scratchpad inventory (2×K, 2×V double-buffered, P, Q,
  output tiles) for `tile_k` against the 1 MiB Gemmini scratchpad budget:
  max ≈1,912 elements exactly, `tile_k`=1024 as the practical choice.
- Caught a second, independent layer of tiling: `tile_k`=1024 is itself far
  bigger than any realistic array physical size (~128–256) — `tile_k` is a
  scratchpad-residency number, not an array dimension; the array sweeps
  through it in further sub-passes, mirroring the original seq_len→array
  relationship one level down.
- Built up systolic-array dataflow mechanics from first principles (which 2
  of a GEMM's M/N/K are spatial vs. temporal defines weight-/output-/
  input-stationary) after getting stuck reasoning about d_head's role by
  analogy alone.
- Applied that framework to both matmuls under the already-chosen K/V-
  stationary dataflow: QK^T and ·V both resolve to the identical spatial
  pair (`d_head`, k-tile), just transposed — resolving the d_head-role
  question and confirming one physical array serves both matmuls without a
  cross-matmul utilization conflict (given an array/feed-routing assumption,
  stated explicitly and carried into Phase 1d).
- Resolved the square-(128×128)-vs-wide-(128×256) array question: no exact
  area/power budget exists to decide it definitively, and array width is
  itself a Timeloop 1c sweep dimension — locked in 128×128 as the primary
  hypothesis (real Trainium precedent) with 128×256 as an explicit alternate
  to test, rather than forcing one answer.
- **Phase 1b fully consolidated**: added a "Final Hypothesis" section (PE
  array, dataflow, scratchpad/accumulator sizing, stated assumptions,
  predictions to check), a "Key Takeaways" section (7 points) for the final
  writeup, and an Open/Not-Yet-Done checklist, mirroring Phase 1a's
  structure. **Phase 1b complete — ready for Phase 1c (Timeloop sweep).**
- Walked the full end-to-end mechanism (HBM Q/K/V → scratchpad K/V-chunk
  fetch → per-group Q streaming → array sub-passes → accumulator →
  scratchpad P → ·V → accumulator → HBM output) to sanity-check the whole
  Phase 1b hypothesis coheres as one pipeline, not just as independently
  correct pieces.
- Caught (via that walkthrough) that "K/V stationary" is scoped to the
  HBM↔scratchpad boundary, not a claim that the array's PE contents never
  change — the array reloads a new 128-wide K-subtile from scratchpad every
  sub-pass (cheap, on-chip), which doesn't contradict K/V being "stationary"
  at the boundary that actually matters for the GQA byte-savings claim. Same
  "stationary is scoped to a memory boundary" pattern recognized
  independently for the third time in this project (scratchpad vs.
  accumulator; tile_k vs. array; now HBM vs. scratchpad).
- Identified an explicit, previously-unstated softmax-granularity choice
  (coarse: one update per full `tile_k` chunk, matching the existing P-tile
  scratchpad sizing; vs. fine: one update per 128-wide array sub-pass,
  smaller P footprint but more control-overhead instructions) — logged as
  coarse-by-choice, with the tradeoff stated rather than hidden.
- **Found a genuinely missing accumulator term**: the raw S/P block
  (pre-softmax QK^T output, matmul-accumulation precision) needs transient
  accumulator residency before softmax converts+requantizes it to the int8
  P that lives in scratchpad — distinct from the partial-output accumulator
  term (they only coincided in size, 32×128, because the primary array
  hypothesis is square 128×128; checked against the 128×256 alternate and
  confirmed they're independent terms). Sized at ≈16 KB (one reused buffer,
  not ×8 heads) — fits into the ~126 KB of existing accumulator headroom
  without changing `tile_q`=32. Updated the Final Hypothesis section
  accordingly.
- Broke the array-level load/evict/reload cycle for one K-subtile down into
  concrete steps (drain to accumulator, no writeback needed since K is
  read-only, load next subtile from scratchpad, re-stream the same
  resident Q-tile) — noted this cycle repeats `8 subtiles × 256 Q-tiles ×
  8 heads` = 16,384 times per scratchpad-resident K/V chunk, all "free" in
  HBM bytes but real array cycles and scratchpad read bandwidth for
  Timeloop's cost model to total up.
- Confirmed softmax touches no array cycles at all — its ops (max,
  subtract, exp, sum, divide, rescale) are elementwise/reduction, not
  matmul, and need a separate vector/scalar functional unit reading/writing
  the accumulator. Flagged as an open Phase 1d question whether Gemmini has
  a native unit for this or requires routing through the host Rocket/BOOM
  core — real potential source of hand-analysis/Timeloop-vs-RTL divergence,
  connects to the Phase 1a "convention for counting exp" open item.
- Checked whether the coarse softmax granularity (assumed since it was
  first introduced) actually fits the accumulator budget once the raw-S
  term is counted honestly: it doesn't — 133,120 B (8-head running state) +
  131,072 B (full `tile_q×tile_k` raw S) = 264,192 B, over the 262,144 B
  budget by 2 KB. This is what actually settled the coarse-vs-fine tradeoff
  flagged earlier, not just a preference — **switched the hypothesis to
  fine-grained** softmax (per 128-wide array sub-pass), which fits with
  large margin and avoids inventing an unanalyzed precision cut, at the
  cost of the already-accepted 8×-more-instructions tradeoff. Also noted
  this converges toward the real Flash-Attention algorithm's structure
  rather than an ad-hoc alternative.
- Re-solved `tile_k` under the now-fixed (not `tile_k`-scaling) P term:
  ceiling nearly doubled to 2,024. But `seq_len_k`=8192=2¹³ means only
  power-of-2 `tile_k` values divide it evenly without a ragged final chunk,
  and the next power of 2 (2048) exceeds 2,024 — so **`tile_k`=1024 stands
  unchanged**, now justified by an evenness/divisibility argument specific
  to this workload's shape rather than a general hardware-friendliness
  heuristic. Phase 1b's final hypothesis updated to reflect all of the
  above (fine-grained softmax, corrected accumulator accounting, `tile_k`
  re-justified).
- Wrote out the full loop nest explicitly, all levels (batch, KV-group,
  Q-tile, K/V-chunk, head, array sub-pass) — not just describing each piece
  in isolation — and this surfaced a major finding: `tile_q`=32 only
  supports one Q-tile's worth of simultaneous accumulator state (the solve
  never had a factor for multiple Q-tiles), which forces Q-tile to be the
  *outer* loop relative to K/V-chunk, which in turn means each K/V-chunk
  gets re-fetched from HBM on the order of 256× (once per Q-tile) rather
  than once per group — breaking the "fetched once per group" GQA reuse
  assumption underlying Phase 1a's compulsory-bytes lower bound.
- Assessed the damage honestly rather than panicking: FLOPs, roofline/AI
  methodology, ridge point, Amdahl's-Law insight, and dataflow/array-shape
  derivation are all unaffected. This is precisely the category of gap
  Phase 1a's own bytes-moved section predicted in advance ("real mappings
  may re-read data if scratchpad can't hold what's needed... a predicted
  source of divergence from Timeloop/Gemmini") — finding a concrete
  instance of it by hand is the prediction paying off, not wasted work.
- Identified a real, unexplored design lever (smaller `tile_q` trading
  instruction count for better K/V reuse) but **deliberately did not
  hand-optimize it further** — logged as an explicit, sharpened prediction
  for Phase 1c instead, consistent with the project's "defensible
  hypothesis, not optimal" framing for Phase 1b.
- Considered raising the accumulator size to fix the re-fetch tension,
  citing TPU v1's real (verified) 4 MB accumulator — caught that this is
  the wrong comparison basis (custom datacenter ASIC vs. a small RTL
  generator, same category of error as the earlier TPU 8t/8i comparison).
  Redirected to the actually-relevant fact instead: Gemmini's
  `acc_capacity` is a configurable generator parameter, not a hardcoded
  256 KB ceiling.
- **Confirmed directly from a real Gemmini paper figure** (user-provided):
  a published, benchmarked "BigSP" SoC config exists with **512 KB
  scratchpad + 512 KB accumulator** (+1 MB L2), alongside a third real
  config ("BigL2," bigger L2 instead of bigger scratchpad/accumulator) —
  settles the "is a bigger accumulator realistic" question with a real
  source, not just plausibility. Also surfaced a real, sobering data point
  from that same figure: BigSP's measured speedup on the **Matmul**
  benchmark category (closest to attention) is small (~1% single-core,
  ~3% dual-core) versus Conv's ~11%/~10% — a caution against assuming
  "bigger accumulator" straightforwardly fixes the re-fetch tension
  proportionally; flagged as a real expectation-calibration point for
  Phase 1c/1d interpretation, not just hand-waved optimism. **Phase 1b
  complete, with two open findings carried into Phase 1c** (K/V-reuse vs.
  `tile_q` tension; accumulator capacity as a free parameter, now backed by
  a confirmed real config point — 512 KB — rather than left fully open).
- Extended the "Key Takeaways (Phase 1b)" section from 7 to 17 points,
  covering everything found since the first takeaways pass: the recurring
  "stationary is boundary-scoped" pattern, the raw-S-vs-partial-output
  distinction, the coarse-softmax-accumulator-overflow discovery, the
  seq_len_k divisibility argument, the major K/V-reuse-vs-accumulator
  finding and the deliberate choice not to hand-optimize past it, the two
  TPU-comparison-basis corrections (v1 and 8t/8i), the BigSP confirmation
  and its calibrating Matmul-vs-Conv data point, and the Phase 1b/1c
  division-of-labor framing. **`notes.md` and `handoff.md` both
  fully up to date — Phase 1b genuinely complete, ready to start Phase 1c.**

---

## Phase 1c: Timeloop Sweep — Recovered State (session-loss incident)

**Context:** Phase 1c was run end-to-end in a separate Claude Code session
(working directly in the `timeloop-accelergy-exercises` docker environment,
not this repo) that was destroyed when the repo got renamed mid-session. The
tool artifacts and Docker container survived on disk; this section
reconstructs the factual record (configs, results, raw mapper output) from
those surviving files so nothing is lost. **This is a factual log, not a
re-derivation** — interpretation of the open findings below is still
outstanding Phase 1c/1d work.

### Artifacts (in `timeloop-accelergy-exercises/workspace/attention_1c/`)

- `problem_qkt.yaml` / `problem_v.yaml` — QK^T and ·V expressed as Timeloop
  conv-style GEMMs. Confirmed: `C·M·N·G = 2^44` MACs for each, matching
  Phase 1a's QK^T/·V FLOPs exactly (sanity check comment in each file).
- `arch.yaml` — Phase 1b's architecture translated into Timeloop: 128×128 PE
  array (sweepable via `run_one.py` args), 1 MiB scratchpad / 256 KiB
  accumulator (sweepable), DRAM at TPU v5e HBM bandwidth. Dataspace
  `keep`/`bypass` deliberately left **unconstrained** at the scratchpad
  level, so the mapper has to discover K/V-stationary on its own rather than
  being told — the actual Phase 1c test of the Phase 1b hypothesis.
- `mapper.yaml` / `run_one.py` — `random_pruned` mapper, optimizing EDP;
  `run_one.py` takes `(problem, out_dir, mesh_x, mesh_y, scratch_depth,
  accum_depth, [search_size, victory_condition, timeout])`.

### Runs completed (real Timeloop output, from `outputs/*/timeloop-mapper.stats.txt` + `.map.txt`)

| run | clock model | kernel | Utilization | Cycles | Winning map: scratchpad-resident dataspace |
|---|---|---|---|---|---|
| `test_qkt` | — | QK^T, 1×1 array (sanity check) | 1.56% | 68,719,476,736 | — |
| `primary_qkt` | 1 GHz, uniform rate | QK^T | **100%** | 1,073,741,824 (=2^30, exact compute-bound ideal) | **none** — Scratchpad level empty in winning map |
| `primary_qkt_v2` | 2 GHz, int8-pumped rate (DRAM BW halved in words/cycle to hold bytes/s fixed — a modeling correction made mid-session) | QK^T | **100%** | 1,073,741,824 | none — same as v1 |
| `primary_v` | 1 GHz | ·V | **100%** | 1,073,741,824 | `Outputs:16384` — mapper chose **output-stationary** |
| `primary_v_v2` | 2 GHz corrected | ·V | 80.63% | 1,331,701,751 | `Weights:1048576` (full 1 MiB) — mapper chose **weight-stationary** (V-stationary, matching the Phase 1b hypothesis) but landed in a worse local optimum than v1's output-stationary result |

DRAM traffic in `primary_qkt`'s winning map: `Weights(K):268,435,456 B` (256
MiB, matches Phase 1a's GQA K bytes exactly) + `Inputs(Q):2,147,483,648 B` (2
GiB) + `Outputs(P):137,438,953,472 B` (128 GiB — one full P-write to DRAM,
i.e. this run models QK^T **unfused**, matching Phase 1a's single P
round-trip term exactly). `primary_v`'s DRAM `Inputs` shows the same
137,438,953,472 B, i.e. this P traffic is read back in for ·V — consistent
with the two kernels being modeled as separate, unfused GEMMs rather than a
fused flash-attention-style pipeline.

### `_v3` runs: bigger search budget, killed before completion

`primary_qkt_v3` / `primary_v_v3` used the identical v2 architecture (128×128,
1 MiB scratch, 256 KiB accum, 2 GHz corrected model) with a much larger
mapper search budget (`search_size=1,500,000`, `victory_condition=30,000`,
`timeout=300,000`, vs. the earlier runs' defaults of ~200,000/5,000/100,000)
— an attempt to get a higher-confidence/more-exhaustive answer, particularly
given `primary_v_v2`'s suboptimal 80.63% result above.

Both jobs ran for **over 2 days** (discovered still running, ~99% CPU each,
when this session started). Per-job, 2 of 4 mapper search threads had
already terminated cleanly (one via the 375,000-valid-mappings cap, one via
the 30,000-stale-mappings victory condition) and their raw `log.txt` already
recorded the same 100%-utilization / 1,073,741,824-cycle mapping found in
`primary_qkt`/`primary_v`. The other 2 threads per job had not yet printed a
termination line after 2+ days.

**Decision made:** killed both jobs (container-internal PIDs, via `docker
exec ... kill`) rather than let them keep running. Rationale: 100%
utilization is the theoretical ceiling for this MAC count/array size — it
was already found — so remaining runtime could only be chasing marginal EDP
(energy) improvements at the same cycle count, not a different or better
answer, and open-ended multi-day runtime for that isn't worthwhile. No
`stats.txt`/`map.txt` exists for `_v3` (the parse step needs all 4 threads to
finish; only raw `log.txt` survives). If a complete `_v3` record is wanted
later for the writeup, rerun with a bounded `search_size` (e.g. 375,000,
matching what thread 0 already completed in the killed runs) so it
terminates in a normal window instead of open-ended.

### Open findings carried forward (Phase 1c/1d interpretation)

1. **RESOLVED — QK^T hits 100% utilization despite the full unfused 128 GiB
   P-write, because the ridge point that matters here isn't Phase 1a's
   480.5.** Phase 1a's ridge point (480.5 FLOPs/byte) was derived from TPU
   v5e's real peak compute (4 MXUs × 128×128 × 1.5 GHz × 2× int8 =
   3.94e14 FLOPs/s) over its real HBM bandwidth. This Timeloop architecture
   models a **single** 128×128 array — not TPU v5e's 4 MXUs — at a 1 GHz
   base clock, not TPU v5e's real 1.5 GHz base clock, while DRAM bandwidth
   is held fixed at the real TPU v5e value throughout. Two independent
   factors, both lowering this architecture's own peak compute relative to
   TPU v5e: **4× from the array count, 1.5× from the base-clock mismatch**,
   for a combined **6×** lower peak compute at the same memory bandwidth →
   this architecture's own ridge point is 480.5 / 6 ≈ **80.08 FLOPs/byte**,
   not 480.5. Unfused AI≈126 is *below* Phase 1a's TPU v5e ridge (480.5) but
   *above* this architecture's own ridge (80.08) — so the correct prediction
   for this specific modeled system was compute-bound all along, exactly
   matching the observed 100% utilization / 2^30-cycle result. Not a mapper
   surprise, not a Phase 1b hypothesis failure — a reminder that the ridge
   point is a property of the *specific accelerator being modeled*, not a
   fixed workload characteristic carried over from whatever reference chip
   Phase 1a happened to use. **General lesson for the writeup**: any time a
   Phase 1a/1b roofline conclusion is checked against a Phase 1c/1d tool
   result, the ridge point has to be recomputed for the actual modeled
   hardware, not reused wholesale from the original workload-source chip —
   this generalizes the same "regime is a design decision, not a fixed
   workload property" theme from Phase 1a's Key Takeaway #1, extended one
   level further: regime is also relative to *which hardware* you're
   asking about.
2. **Still open — ·V's winning dataflow was qualitatively different between
   v1 and v2** (output-stationary @ 100% util vs. weight-stationary @
   80.63% util) even though the only thing that changed between those two
   runs was the clock-rate/DRAM-bandwidth modeling correction, not the
   architecture or hypothesis being tested.

   **RESOLVED (mechanistically), via direct comparison of the identical
   mapping across v1/v2 rather than the two missing counterfactual runs
   (weight-stationary forced in v1, output-stationary forced in v2) —
   turned out not to be needed.** The exact same loop-nest mapping was
   independently sampled in both runs' raw search logs:

   ```
   v1 (1 GHz):  L4[WIO] C2 G128 N1024 - L3[W] C32 G2 N32 - ...   Util=1.00, Cycles=1,073,741,824
   v2 (2 GHz):  L4[WIO] C2 G128 N1024 - L3[W] C32 G2 N32 - ...   Util=0.81, Cycles=1,331,701,751
   ```

   Identical mapping, worse cycle count under v2 — a **real, mapping-invariant
   effect**, not search noise: this schedule's DMA time fully hid behind
   compute under v1's clock model and stopped fully hiding once compute got
   2× faster against the same fixed DRAM bandwidth (the same ridge-doubling
   from finding #1, now shown to have a second, distinct consequence: it
   doesn't just reclassify compute-vs-memory-bound, it can break a
   specific mapping's DMA/compute overlap outright).

   Second half of the explanation: in v1's own log, weight-stationary and
   output-stationary were essentially **tied** at 100% utilization
   (pJ/Compute 2.821 vs. 2.820, with a third output-stationary sample at
   2.810 edging out as the reported winner by under 0.5%) — so in v1,
   dataflow choice barely mattered; either one reaches the ideal. In v2's
   own default-budget log (50,000 valid mappings, same budget as v1 used),
   the best output-stationary sample found only reached **50% utilization**
   (Cycles=2,147,483,648) — nowhere near competitive with weight-stationary's
   80.63%. So weight-stationary "won" v2's race not because it was good, but
   because v2's search never happened to sample a competitive alternative —
   even though one demonstrably exists under v2's own architecture (see
   below).

   **Verification that v2's true ceiling is still 100%, not 80.63%:** the
   killed `_v3` run (same v2 architecture, `search_size=1,500,000`) had
   already recorded multiple `Utilization=1.00, Cycles=1,073,741,824`
   samples (tagged `L3[O]`/`L3[WO]`) in its raw `log.txt` before being
   killed — proof the 100%-utilization ceiling is reachable under v2's
   architecture, just not by every mapping, and not within default search
   depth.

   **Follow-up: reran `primary_v_v2` fresh (`primary_v_v2_rerun`, identical
   architecture and default search budget, new output dir) expecting the
   unseeded `random_pruned` search to land somewhere different by chance.**
   It didn't — the new run's `log.txt` is **byte-for-byte identical** to the
   original `primary_v_v2` run, same sample sequence, same thread
   assignments, same final answer (`Utilization=0.81, Cycles=1,331,701,751`,
   weight-stationary). **This means the mapper's `random_pruned` search is
   deterministic given the same config/budget, not randomly seeded per
   invocation** — a real, useful finding in its own right: v2's 80.63%
   result isn't bad luck that a rerun would fix, it's *the* answer this
   search depth always converges to. Reaching the true 100% ceiling requires
   materially more search depth, not another attempt at the same depth.

   **Decision: did not pursue a deeper bounded search further.** From the
   killed `_v3` log, the 100%-utilization sample only appeared close to a
   thread's *entire* 375,000-mapping quota (`search_size=1,500,000` ÷ 4
   threads) — the same depth that caused the earlier multi-day runaway, with
   no evidence a moderate middle-ground budget would reach it reliably.
   Given the mechanism is already fully explained (ridge-doubling breaks the
   specific v1-winning schedule; a different schedule still hits 100% under
   v2 but needs deep search to find; default search is deterministic, not
   unlucky), a clean formal 100%-utilization number for v2's ·V was judged
   not worth chasing further — the "deterministic local optimum requiring
   deep search to escape" result is itself a legitimate, citable Phase 1c
   finding about the mapper tool's own behavior, not a gap needing to be
   closed.

### Third finding: this Phase 1c setup structurally cannot model "fused" at all

Noticed while reviewing the DRAM traffic lines across every run: QK^T's
`Outputs` and ·V's `Inputs` both show the identical 137,438,953,472 B (one
full P round-trip) in *every* run — v1, v2, every dataflow the mapper found.
Not a mapping/dataflow effect: `problem_qkt.yaml` and `problem_v.yaml` are
two independent Timeloop problems, each with its own DRAM-level I/O, with no
on-chip path connecting one kernel's output to the next kernel's input. **No
mapper search, however deep, can ever produce a fused result here** — the
limitation is in the problem formulation, not the search.

This matters because Phase 1b's entire scratchpad/accumulator sizing
exercise (`tile_q`, `tile_k`, online-softmax accumulator state) was designed
specifically to *enable* fusion (P never touching HBM, flash-attention
style) — meaning **Phase 1c as built has never actually exercised the
regime Phase 1b's hardware hypothesis was built around.** Findings #1 and
#2 above are both fully valid, but both are characterizing the *unfused*
regime only.

**Decision: not worth building a true fused Timeloop model.** Vanilla
Timeloop's problem format is built around mapping one GEMM/conv onto a
memory hierarchy — representing "two matmuls with an intermediate tensor
that never leaves the chip" isn't a native capability; forcing it would
mean hacking the architecture model (e.g. artificially zeroing P's DRAM
cost) rather than getting a real answer. A real Gemmini RTL pipeline can
*literally* keep P in scratchpad between the two matmuls — no
fusion-modeling hack needed, just the loop nest as written. So: **fusion
validation is being deferred to Phase 1d**, where it's a natural
consequence of the RTL rather than something Timeloop needs to fake.

**Open footnote for the writeup**: if a rough "fused-equivalent" number is
wanted before Phase 1d lands, it can be hand-projected cheaply — subtract
the shared P-traffic bytes/energy from the existing QK^T and ·V Timeloop
results using Phase 1a's own byte formulas, rather than rerunning anything.
Not yet computed; logged here as a lightweight stopgap option, not a
substitute for Phase 1d's real validation.

---

## Key Takeaways (Phase 1c) — for final writeup

1. **The ridge point is a property of the specific accelerator being
   modeled, not a fixed workload characteristic.** Phase 1a's TPU
   v5e-derived ridge point (480.5) doesn't transfer to a deliberately
   smaller single-array Timeloop/Gemmini model — this architecture's own
   ridge (≈80.08, from 6× lower peak compute at the same real HBM
   bandwidth) is what actually determines its compute-vs-memory-bound
   regime. Any time a Phase 1a/1b roofline conclusion is checked against a
   Phase 1c/1d tool result, the ridge point has to be recomputed for the
   actual modeled hardware.
2. **How much dataflow choice matters is itself regime-dependent.** Near/
   below an architecture's own ridge point (v1, easy to saturate),
   competing dataflows can be essentially fungible — weight- and
   output-stationary tied at 100% utilization within a rounding error. Once
   an architecture change pushes the ridge higher (v2, harder to saturate),
   the same mapping that used to tie for optimal can fall far behind, and
   which dataflow "wins" starts to matter a great deal more. Dataflow
   sensitivity isn't a fixed property of the workload — it depends on how
   much headroom the architecture has relative to its own ridge.
3. **A single mapper run's "winner" is not necessarily the architecture's
   true optimum — it's whatever a deterministic, budget-limited search
   happens to find.** Confirmed directly: rerunning the identical config
   produced a byte-for-byte identical search log and the same answer.
   Escaping a suboptimal result requires more search depth (or narrowing the
   space), never just another attempt at the same depth — a methodological
   lesson for interpreting *any* Timeloop mapper result, not just this one.
4. **A tool's structural modeling limitations can silently scope out the
   exact regime you built the hardware hypothesis to test.** This Phase 1c
   setup (two independent per-kernel Timeloop problems) can only ever
   represent the unfused case, no matter how the architecture or mapper
   search is varied — yet fusion was the entire premise behind Phase 1b's
   scratchpad/accumulator sizing. Worth checking, for any tool used in this
   project, whether its native problem representation can even express the
   hypothesis being tested before trusting its results as validation of it.

**Phase 1c status: both open findings from the recovered results are
resolved at the mechanistic level (finding #1: ridge point is
architecture-specific, not carried over from Phase 1a's TPU v5e reference;
finding #2: real ridge-doubling effect on a specific mapping, compounded by
a deterministic default-search limitation, not true search-variance/luck),
plus a third structural finding (this setup cannot model fusion at all,
deferred to Phase 1d rather than hacked around in Timeloop). Phase 1c's
comparison-and-gap-explanation step is complete. Next milestone per the
spec: Phase 1d (configure Gemmini, read the generated RTL, validate via
Verilator).**
