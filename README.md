# Attention Hardware-Software Co-Design

*README generated with [Claude Code](https://claude.com/claude-code)*

Workload → hardware codesign for LLM attention: hand-derived roofline analysis of prefill attention, validated into a concrete systolic-array hardware hypothesis, with every prediction logged before being checked. **Status: Phase 1 (prefill) hand-analysis and hardware hypothesis complete; Timeloop/Gemmini-RTL validation (Phase 1c/1d) and the decode-regime comparison (Phase 2) are in progress** — this is a self-directed project, not a course assignment, and this README reflects exactly where it stands today rather than the finished scope.

**Stack:** Timeloop/Accelergy, Gemmini (Chipyard), Verilator · **Workload:** Llama 3-70B prefill attention (GQA), from *How to Scale Your Model*'s TPU-serving numbers

- **Proved the compute/memory-bound regime is a design decision, not a workload property**: the exact same workload shape hits AI = 8192 FLOPs/byte (decisively compute-bound, fused) or AI ≈ 126 (decisively memory-bound, unfused) against a TPU v5e ridge point of ≈480.5 — the fusion decision alone swings the regime 65×.
- **Showed GQA's 8× KV-cache reduction is not an 8× bandwidth win**: total compulsory bytes only drop 1.78× (Amdahl's Law — Q/output are invariant to KV-head count and floor the total), and in the fused/compute-bound regime GQA doesn't reduce execution time *at all* since time is set by FLOPs, which GQA leaves untouched.
- **Derived a full hardware hypothesis from first principles** (128×128 systolic array matching Trainium precedent, K/V-stationary weight-stationary dataflow, `tile_k`=1024, `tile_q`=32) and, while stress-testing it, independently discovered the tension that makes GQA's on-paper reuse hard to realize on real hardware (see below) — before writing a single line of Timeloop/Gemmini config.

## Methodology

Every phase follows the same loop: **hand-derive a prediction → search the design space with a tool → validate in RTL → explain every gap mechanistically.** The two hand-derivation phases (1a: roofline, 1b: hardware hypothesis) are done for prefill; the tool-validation phases (1c: Timeloop sweep, 1d: Gemmini/Verilator) are next.

```
1a  hand-derive FLOPs, bytes, arithmetic intensity, ridge point   ✅ done
1b  hand-derive PE array / dataflow / scratchpad / accumulator   ✅ done
1c  sweep the same design space in Timeloop, compare to 1b       ⏳ next
1d  configure Gemmini, read the generated RTL, validate in       ⏳ not started
    Verilator, explain every gap vs. Timeloop
```

## Phase 1a: Roofline analysis

Workload: batch=32, seq_len=8192, n_heads=64, n_kv_heads=8 (GQA), d_head=128, int8. Derived FLOPs and bytes-moved for two variants (naive MHA as a control, then GQA) and two fusion assumptions (does the softmax/P-matrix intermediate ever touch HBM):

| | FLOPs | Bytes (fused) | Bytes (unfused) | AI (fused) | AI (unfused) |
|---|---|---|---|---|---|
| MHA | 2⁴⁶ | 8 GiB | ≈520 GiB | 8,192 | ≈126 |
| GQA | 2⁴⁶ (identical) | 4.5 GiB | ≈516.5 GiB | ≈14,564 | ≈126.9 |

Ridge point (TPU v5e, int8, same source as the workload): **≈480.5 FLOPs/byte**. Both fused and unfused bounds sit decisively on opposite sides of it — this isn't a close call sensitive to modest ridge-point error.

Two findings that came out of comparing the MHA and GQA rows rather than looking at either alone:

- **The regime is set by fusion, not workload shape.** Whether this workload is compute- or memory-bound depends almost entirely on whether the P-matrix round-trips to HBM (a 65× AI swing), not on batch/seq_len/head count.
- **GQA's payoff is Amdahl's-Law-shaped, twice over.** (1) The 8× K/V byte reduction barely moves total AI when unfused, because P-traffic — untouched by KV-head sharing — dominates the unfused total by ~100×; it only shows up in full once fusion removes P-traffic. (2) Even in the fused/compute-bound regime, GQA doesn't reduce execution time at all, since compute-bound time is set by FLOPs (unchanged by GQA) — its real fused-regime payoff is scratchpad pressure and KV-cache footprint, not throughput. The throughput win is a decode-regime (Phase 2) prediction, not yet checked.

## Phase 1b: Hardware hypothesis

Resolved bottom-up from a first-principles systolic-array framework (a systolic array makes exactly 2 of a GEMM's M/N/K dims spatial; the dataflow name is the choice of which pair):

- **PE array:** 128×128 (matches real Trainium precedent), with 128×256 carried as an explicit alternate — no exact area/power budget exists to pick between them by hand, and array width is itself one of Timeloop's Phase 1c sweep dimensions.
- **Dataflow:** weight-stationary, K/V as the stationary operand — driven directly by the 8× reuse factor (`num_q_heads / num_k_heads`) across the group of query heads sharing one KV head. The same physical array serves both QK^T and ·V because `d_head` anchors the spatial pair in both matmuls.
- **Scratchpad** (≤1 MiB, Gemmini default): `tile_k`=1024 — solved against the full scratchpad inventory (double-buffered K/V chunks, P tile, Q tile, output tile), and pinned to 1024 specifically because `seq_len_k`=8192=2¹³ only divides evenly by power-of-2 tile sizes.
- **Accumulator** (≤256 KiB): `tile_q`=32 — solved against per-head online-softmax tracking state (running max, running sum, partial-output accumulator) for all 8 heads sharing one KV head simultaneously.

**The standout finding** came from walking the full loop nest explicitly instead of reasoning about each level in isolation: `tile_q`=32 was sized assuming only *one* Q-tile's accumulator state resident at a time, which forces Q-tile to be the outer loop relative to K/V-chunk — and that ordering means each K/V-chunk gets re-fetched from HBM ~256× (once per Q-tile) instead of once per group, **breaking the "fetched once" GQA reuse assumption Phase 1a's own byte counts were built on.** This isn't a broken hypothesis; Phase 1a explicitly flagged this exact category of gap in advance ("real mappings may re-read data if scratchpad can't hold what's needed"). It's left as a sharpened, falsifiable question for Phase 1c: does Timeloop's mapper converge to `tile_q`=32 anyway (accepting the re-fetch), or find a smaller tile that trades instruction count for reuse?

A related question — whether a bigger accumulator (Gemmini's `acc_capacity` is configurable, not fixed at 256 KB) fixes the tension — is grounded in a real published Gemmini config (**BigSP: 512 KB scratchpad + 512 KB accumulator**) rather than speculation, but that same config's measured speedup on Matmul-category workloads (~1–3%) versus Conv (~10–11%) is logged as a reason not to assume it proportionally helps.

## What's next

- **Phase 1c:** build the Timeloop/Accelergy workload + architecture + mapper specs and sweep array shape, dataflow, and memory sizing against the Phase 1b hypothesis above.
- **Phase 1d:** configure Gemmini to the Timeloop-derived config, read the generated RTL, and validate cycle counts through Verilator — explaining every gap against Timeloop mechanistically.
- **Phase 2:** repeat the full 1a→1d loop for decode-phase attention (memory-bound by construction) and compare the "ideal hardware" conclusions against prefill's.
- **Phase 3:** precision/numerics tradeoffs (int8 vs. higher precision), tied to prior FP8/MXFP8 kernel-porting work.

Full derivation trail (every hypothesis, correction, and dead end, not just the conclusions) is in [`phase1_notes.md`](phase1_notes.md).
