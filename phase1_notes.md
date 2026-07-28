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
