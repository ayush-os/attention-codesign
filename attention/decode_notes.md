# Decode Attention — Workload → Silicon (Phase 2, In Progress)

Live working log for Phase 2 (decode-phase attention, memory-leaning regime), Prediction/Log style — matches how Phase 1's derivation was tracked before being polished into `prefill_notes.md`. This file will be refactored into a finished writeup at Phase 2's natural completion point per `handoff.md`; nothing here should be treated as final/polished yet.

---

## 0. Workload and Scope

**Workload**: same Llama 3-70B GQA config as Phase 1, chosen for continuity (not re-derived from scratch, a stated choice):

| Parameter | Value | Note |
|---|---|---|
| batch | 32 | same as prefill — see §2, found inert for AI given project scope |
| seq_len_q | 1 | new: prefill's single `seq_len` splits into two params for decode |
| seq_len_kv | 8192 | "running length of the convo" — an already-filled context, same magnitude as prefill's `seq_len` for continuity |
| n_heads (query) | 64 | unchanged |
| n_kv_heads (GQA) | 8 | unchanged |
| d_head | 128 | unchanged |
| precision | int8 | unchanged |

**Scope note**: this project's decode analysis covers SDPA only (QK^T → softmax → ·V) — not the surrounding QKVO/FFN projection weights. This matters directly for §2 (batch) below: weight-amortization-via-batching is a real decode phenomenon, but it lives in the projection weights (shared across batch), not in K/V (batch-unique data) — out of scope here by construction, not an oversight.

---

## 1. Matrix Dimensions (naive SDPA, per batch element & head)

Holds for both prefill and decode — only the `seq_len_q`/`seq_len_kv` values differ:

| Matrix | Shape |
|---|---|
| Q | `(seq_len_q, d_head)` |
| K | `(seq_len_kv, d_head)` |
| V | `(seq_len_kv, d_head)` |
| S = QKᵀ | `(seq_len_q, seq_len_kv)` |
| P = softmax(S) | `(seq_len_q, seq_len_kv)` |
| O = PV | `(seq_len_q, d_head)` |

At decode's `seq_len_q = 1`: Q → `(1, 128)`, S/P → `(1, 8192)`, O → `(1, 128)` — Q, S/P, and O all collapse to vectors.

---

## 2. FLOPs

**QK^T**: output `(seq_len_q, seq_len_kv)`, contraction dim `d_head` → FLOPs = `2 × seq_len_q × seq_len_kv × d_head`.
**PV**: output `(seq_len_q, d_head)`, contraction dim `seq_len_kv` → FLOPs = `2 × seq_len_q × d_head × seq_len_kv`.

**Correction to Phase 1's framing**: `prefill_notes.md` §1.1 attributed QK^T = PV FLOPs to "seq_len appearing symmetrically either way, for *this particular shape*" (i.e. `seq_len_q = seq_len_kv` in prefill). Checked here: the two expressions above are the same three-way product (`seq_len_q × seq_len_kv × d_head`) written in different order — multiplication is commutative, so **this equality is a structural identity of SDPA's two matmuls, true for any `seq_len_q`/`seq_len_kv`, not a coincidence specific to prefill's equal-lengths shape.** Confirmed here where `seq_len_q ≠ seq_len_kv` and the equality still holds. Worth correcting this language if `prefill_notes.md` is ever revised.

**Per (batch, head)**: QK^T = PV = `2 × 1 × 8192 × 128` = 2,097,152 = 2²¹ FLOPs each.

**Total, scaled by batch × n_heads (2¹¹)**: Total FLOPs (QK^T + PV) = **2³³ ≈ 8.59 × 10⁹**.

**GQA and compute**: confirmed unchanged from Phase 1a's finding — GQA is a bytes-moved lever only, doesn't touch FLOPs, in either regime. FLOPs count query-head *executions*, not how many distinct KV tensors back them; this transfers directly from prefill, doesn't need re-deriving per regime. **Total FLOPs = 2³³ for both MHA and GQA.**

---

## 3. Bytes Moved

**Fusion is not a meaningful lever in decode** — a real, load-bearing difference from prefill. P/S is now `(1, 8192)` ≈ 8 KiB (int8), trivially SRAM-resident regardless of fusion choice — unlike prefill, where P's 64 MiB-per-head size *forced* tiling and made fused-vs-unfused a 65× AI swing. In decode, P never becomes large enough to be the bottleneck either way, so there's no "unfused" regime worth deriving as a second bound the way Phase 1a did for prefill.

**Per (batch, head), MHA** (no cross-head K/V sharing): load = `Q(128 B) + K+V(2 × 8192 × 128 B) = 128 + 2,097,152 = 2,097,280 B`; write (O) = `128 B`. Total = 2,097,408 B/unit.

**Scaled by batch × n_heads (2048)**: **MHA total = 4,295,491,584 B ≈ 4.0005 GiB.**

**GQA** (K/V loaded once per KV-head group, not per query head — same "compulsory bytes, perfect on-chip reuse" idealization Phase 1a used for prefill; whether a real scratchpad can hold a full K/V head resident across the 8-head group is a Phase 2b question, not this one):

Per batch: load = `n_heads×128 (Q) + n_kv_heads×8192×128×2 (K+V)`; write = `n_heads×128 (O)`.
**Scaled by batch (32): GQA total = 537,395,200 B ≈ 0.5005 GiB.**

**MHA/GQA ratio ≈ 7.99×** — essentially the *full* theoretical 8× group-size reduction. Contrast with prefill's capped **1.78×** (`prefill_notes.md` §1.2): prefill's total was capped by Q/output being large and structurally invariant to `n_kv_heads` (Amdahl's-Law-style capping). In decode, Q and O have collapsed to `(1, d_head)` — negligible relative to K/V — so that capping term is gone and GQA's local 8× win passes through to the total almost fully intact.

---

## 4. Arithmetic Intensity and Ridge Point

| | AI (FLOPs/byte) |
|---|---|
| MHA | 2³³ / 4,295,491,584 ≈ **2.0** |
| GQA | 2³³ / 537,395,200 ≈ **15.98** |

**Ridge point**: 480.5 FLOPs/byte (TPU v5e, int8 — same reference chip as Phase 1a, for internal consistency).

**Both decisively memory-bound** — not a close call: MHA is **~240× below ridge**, GQA-improved is **~30× below ridge**. Far more decisive in either case than prefill's unfused regime (~3.8× below ridge). Decode is structurally, unavoidably memory-bound at this shape — there's no analog to prefill's fusion lever that could flip it.

**GQA's payoff resolves Phase 1a's pre-registered open thread directly** (`prefill_notes.md` §1.4 Key Takeaway #7 / §6): fused prefill got *zero* throughput benefit from GQA (FLOPs-bound, GQA doesn't touch FLOPs). Decode just showed the real payoff — an 8× AI jump (2 → 16) that reflects directly in real bytes-moved, because there's no Amdahl's-Law-capping term here to blunt it.

---

## 5. Cross-Phase Comparison: Prefill vs. Decode (FLOPs & Bytes)

Direct comparison against `prefill_notes.md`'s numbers, to make explicit *why* the two regimes end up on opposite sides of the ridge point by such different margins.

**FLOPs:**

| | Prefill | Decode | Ratio |
|---|---|---|---|
| MHA & GQA (identical within each phase) | 2⁴⁶ ≈ 7.037×10¹³ | 2³³ ≈ 8.59×10⁹ | prefill is **2¹³ = 8,192×** more |

**Bytes** (decode compared against prefill's *fused* numbers — the fair comparison, since decode has no meaningful unfused case, §3):

| | Prefill (fused) | Decode | Ratio |
|---|---|---|---|
| MHA | 8 GiB | ≈4.0005 GiB | prefill is **~2×** more |
| GQA | 4.5 GiB | ≈0.5005 GiB | prefill is **~9×** more |

**The mechanism, precisely (not just "roughly from seq_len_q")**: K and V bytes are *literally identical* between prefill and decode — both depend only on `seq_len_kv` (=8192, unchanged across regimes), never on `seq_len_q`. The entire prefill-vs-decode byte gap is 100% attributable to Q and output, the only two tensors that scale with `seq_len_q` (8192 in prefill → 1 in decode, collapsing to near-zero). Checked exactly: prefill MHA fused = Q(2GiB)+K(2GiB)+V(2GiB)+O(2GiB) = 8 GiB; decode MHA ≈ K(2GiB)+V(2GiB)+negligible Q/O ≈ 4 GiB — the 2× gap *is* Q+O, exactly. Same for GQA: prefill's 4.5 GiB = Q(2)+K(0.25)+V(0.25)+O(2); decode's 0.5 GiB ≈ K(0.25)+V(0.25) alone.

**Core intuition, tying FLOPs and bytes together**: `seq_len_q` is the knob that decides how many FLOPs get to share (amortize) each byte of K/V fetched. Prefill cranks it to 8192, decode collapses it to 1 — same K/V-fetch bill in both cases, but prefill gets to spread that cost over 8,192× more compute. This *is* arithmetic intensity: "FLOPs amortized per byte moved." It's the whole reason prefill can land 17× above the ridge point while decode lands 240× below it, off the same underlying K/V access pattern.

---

## 6. Key Findings So Far (Phase 2a)

1. **Decode splits prefill's single `seq_len` into two independent params** (`seq_len_q = 1`, `seq_len_kv` = context length) — not a substitution, a structural change to the workload's shape.
2. **QK^T = PV FLOPs is a general structural identity** (commutative product), not a coincidence of prefill's equal-length shape as originally framed — corrects, doesn't overturn, `prefill_notes.md` §1.1.
3. **Batch is inert for SDPA's own AI**, by construction: every batch element carries a distinct KV cache, so FLOPs and bytes both scale linearly with batch — no cross-batch reuse to amortize, unlike the (out-of-scope) FFN/projection weights. Batch=32 kept for continuity, not because any derivation favors it over another value.
4. **Fusion, the dominant lever in prefill (65× AI swing), is not a meaningful lever in decode** — P/S is ~8 KiB, trivially on-chip regardless of fuse/unfuse choice. Decode only has one regime worth deriving, unlike prefill's fused/unfused pair.
5. **GQA's byte-savings pass through almost fully in decode (~7.99× of the theoretical 8×)**, vs. prefill's Amdahl's-Law-capped 1.78× — because Q/output collapsed to negligible size, removing the capping term that limited prefill's win.
6. **GQA is now first-order, not secondary.** In prefill, GQA only mattered once fusion (the dominant lever) was already solved (`prefill_notes.md` §1.4 Key Takeaway #4). In decode, with no fusion lever available at all, GQA is the *only* lever within SDPA itself that moves AI — a genuine regime-driven role reversal for the same technique. (Open aside, not pursued: sparse/linear attention would introduce additional levers, out of this project's current scope.)
7. **Decode is far more decisively memory-bound than prefill's memory-bound case ever was** (~240×/~30× below ridge vs. prefill unfused's ~3.8×) — the two regimes aren't just opposite, they're opposite by very different margins.

---

## 7. Open / Next

- **Phase 2a is otherwise complete**: FLOPs, bytes (MHA + GQA), AI, ridge comparison, regime call all derived and logged above.
- **Next: Phase 2b** — hardware hypothesis (PE array shape, dataflow, scratchpad/accumulator sizing) for this decisively memory-bound, GQA-first-order profile. Expect this to diverge substantially from Phase 1b's prefill hypothesis (128×128 WS array sized around `d_head` and fused online-softmax tiling) given decode's very different reuse pattern (vector-matrix, not matrix-matrix) and the absence of any P-tiling pressure.
- **Carried over from Phase 1's open threads** (`prefill_notes.md` §6), now partially resolved: "GQA's real throughput payoff in a memory-bound regime" — resolved above (§4, §5). Still open: whether decode's GQA byte-savings are actually *achievable* under a real scratchpad budget (mirrors Phase 1b's `tile_q`-vs-K/V-reuse tension for prefill) — deferred to Phase 2b.
