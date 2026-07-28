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
