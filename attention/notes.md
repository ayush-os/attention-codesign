# Phase 3 (Numerics, Reframed) — Live Derivation Log

Prediction/Log style, per the project's standing convention. Reframed question (per `decode_notes.md` §4, not the spec's literal "compare precision throughput on Gemmini," which would both be generic and hit the Gemmini-representability wall already found in 2b): **is KV-cache quantization (below int8) a bigger lever on decode's AI than GQA was?**

---

## Prediction, before deriving

Lowering K/V storage precision (e.g. int8 → int4) should halve bytes moved. Whether AI actually improves relative to the ridge point depends on a fork not obvious at first: does the compute engine's *peak* FLOPs/s also change, or does only the workload's bytes change?

## Derivation

**First pass — native low-precision compute**: if K/V is both stored *and* computed natively at half the bit-width, two things move together: workload bytes halve (AI doubles), but real hardware's peak FLOPs/s also roughly doubles for lower precision (more low-precision ALUs pack into the same silicon — the same mechanism behind e.g. H100's FP8 TFLOPS dwarfing its FP32 number). Ridge point = peak FLOPs/s ÷ peak bandwidth, so ridge doubles too. **Result: `AI/ridge` is unchanged** — `(2×AI)/(2×ridge) = AI/ridge` — no matter how many precision steps down you go. A real, clean cancellation: in this scenario, numerics is *not* a lever on decode's regime at all, despite genuinely halving bytes moved.

**Second pass — the realistic case, dequant-before-compute**: real KV-cache quantization decouples storage precision from compute precision — K/V is compressed for storage/bandwidth, then dequantized to a higher precision before the actual dot product (attention scores are numerically sensitive; this mirrors the exact pattern Phase 1a already found for a different tensor, `prefill_notes.md` §1.2 — P requantized to int8 for HBM writes, but softmax math itself runs at higher precision on-chip). Under this model, the compute engine never leaves its baseline precision (int8) — **ridge stays fixed at 480.5**. Only bytes move, so the lever is real again, bounded only by how far storage precision can realistically go.

**Quantifying the crossover** (mirrors the MoE project's own `≈2.5 bytes/element` crossover finding): AI scales inversely with bytes/element, holding ridge fixed. GQA's current AI ≈ 16 at 1 byte/element (int8). Solving `16 × (1/bytes_new) = 480.5` → **`bytes_new = 16/480.5 = 1/30 ≈ 0.0333 bytes/element ≈ 0.27 bits/element`.**

**Verdict**: not a realistic quantization target — sub-1-bit-per-element isn't a standard precision format at all, and even the most aggressive real quantization research doesn't approach this for KV caches specifically, given how numerically sensitive attention scores are to K/V precision loss. **Decode attention has a hard, quantization-proof floor**: no realistic (or even wildly unrealistic) K/V precision choice can flip it out of the memory-bound regime.

## Cross-Project Synthesis (the actual portable lesson)

Direct, striking contrast with this repo's MoE project: there, numerics *was* the dominant, regime-flipping lever, with a real crossover sitting in a realistic range (FP8→BF16, `≈2.5 bytes/element`). Here, the same kind of lever exists structurally (bytes genuinely halve per precision step) but is irrelevant to the regime question.

**The generalizable finding, not specific to either workload**: it isn't "does a numerics lever exist" — it's **how large the roofline margin is that the lever has to close**. MoE's worst-case margin was only `~5–6.7×` (`≈21,065` vs. `≈4,208` ridge, even under imbalance, per the MoE README) — well within what a realistic ~2× byte-format swing can plausibly close. Decode's margin is `30–240×` — an order of magnitude (or two) further out than any believable precision format range could ever reach. Same mechanism, same category of lever, opposite verdict, purely a function of the margin size at the workload's starting point — nothing special about attention vs. MoE routing specifically.

## Key Findings

1. Numerics as a lever on regime depends critically on whether compute precision is coupled to storage precision — coupled (native low-precision compute) cancels out via a matching ridge-point shift; decoupled (dequant-before-compute, the realistic case) leaves ridge fixed and the lever intact.
2. Quantified: decode's actual crossover point (`1/30 bytes/element`) is not a realistic target by a wide margin — a clean, hard-floor finding, structurally similar in form to the MoE project's own "hard, imbalance-proof floor" language, but for precision instead of routing skew.
3. **The portable lesson**: whether a numerics lever can flip a regime is a question about the *size of the margin it has to close*, not about whether the lever exists in principle — directly explains why the same technique was decisive for MoE and structurally irrelevant for decode, without needing any difference in mechanism between the two workloads.
