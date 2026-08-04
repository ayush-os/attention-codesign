# Workload → Silicon: Hardware & System Codesign for LLM Inference

Two self-directed codesign studies — single-chip attention microarchitecture, then multi-chip MoE routing — every prediction hand-derived and logged *before* being checked against a tool or the literature.

**Stack:** Timeloop/Accelergy, Gemmini (Chipyard), Verilator, ASTRA-sim · **Workloads:** Llama 3-70B attention (GQA), prefill and decode · DeepSeek-V2 MoE routing (multi-chip, 8-way EP)

- **The compute/memory-bound regime is a design decision, not a workload property**: the same prefill attention workload hits **AI = 8,192 FLOPs/byte** (compute-bound, fused) or **AI ≈ 126** (memory-bound, unfused) against a TPU v5e ridge point of **≈480.5** — the fusion decision alone swings the regime **65×**.
- **Independently found a GQA-breaking bug by hand**, before touching Timeloop or Gemmini: a naive accumulator-driven `tile_q=32` forces Q-tile-outer loop ordering in prefill, causing **~256× K/V re-fetching from HBM** — breaking GQA's textbook "fetch once" reuse claim.
- **Hand-found a second hardware-breaking issue, this time in decode**: a systolic array's fill/drain amortization needs a deep temporal stream — prefill gets 65,536 cycles per K/V load, decode collapses to just 8, landing a naive utilization estimate right at the edge of decode's own ~30× roofline margin. Forced a pivot to a SIMD compute primitive, then confirmed by reading Gemmini's actual source that **no such compute path exists in the generator at all** — a harder tool-representability gap than prefill's own fusion-modeling limit.
- **The same "just quantize it" lever is decisive for the MoE project below and structurally powerless for decode attention** — decode's regime-flipping crossover would need **~0.27 bits/element**, not a realizable format, vs. MoE's real crossover at **≈2.5 bytes/element** (FP8/BF16). Same mechanism, opposite verdict: purely a function of how large a margin each workload's numerics lever has to close (~5–7× for MoE, 30–240× for decode), not any difference in how the lever works.
- **Derived DeepSeek-V2's real dispatch/combine traffic** against the paper's actual device-limited routing mechanism (not the textbook formula) and proved the workload sits on a **hard, imbalance-proof compute-bound floor** (~21,065 FLOPs/byte, ~5× the TPU 8i ICI ridge point) — no routing skew, however severe, can flip it comms-bound. Only a numerics choice (dispatch precision) can.

---

## Methodology (shared across both projects)

Every phase follows the same loop: **hand-derive a prediction → validate against a tool or real literature → explain every gap mechanistically.** Applied first at the single-accelerator level (PE array, dataflow, scratchpad), then one level up at the system level (interconnect topology, bandwidth, buffering). Scope is treated as a first-class, discussable decision throughout, not just something to push through — several phases were deliberately narrowed or skipped once their marginal learning value was checked against real cost, with the reasoning stated rather than left implicit (see both writeups' scope sections).

---

## Attention Hardware Codesign → [`prefill_notes.md`](prefill_notes.md) · [`decode_notes.md`](decode_notes.md)

**Stack:** Timeloop/Accelergy, Gemmini (Chipyard), Verilator · **Workload:** Llama 3-70B attention (GQA), from *How to Scale Your Model*'s TPU-serving numbers — prefill (compute-leaning) and decode (memory-leaning)

**Prefill:**
- Proved the compute/memory-bound regime is a **design decision, not a workload property** (see above) — a 65× AI swing from fusion alone, at a fixed ridge point.
- Showed GQA's 8× KV-cache reduction is **not** an 8× bandwidth win: total compulsory bytes only drop **1.78×** (Amdahl's Law — Q/output activity floors the total), and in the fused/compute-bound regime GQA doesn't reduce execution time *at all*, since time is set by FLOPs, which GQA leaves untouched.
- Derived a full hardware hypothesis from first principles (**128×128** systolic array, K/V-stationary weight-stationary dataflow, `tile_k=1024`, `tile_q=32`) and, stress-testing it by hand, found the `tile_q=32` re-fetch tension above — a sharpened, falsifiable prediction for the Timeloop sweep, not a fixed conclusion.
- Swept the hypothesis in Timeloop and found the ridge point itself doesn't transfer across chips: a small single-array model has **6× lower peak compute** than TPU v5e at the same real HBM bandwidth, so its ridge point is **≈80, not 480.5** — flipping what "unfused" predicts for *this* hardware specifically. Also caught the mapper tool's own blind spot: its search is deterministic (confirmed via a byte-for-byte-identical rerun), so a discovered "optimum" can just be a local one a shallow search never escaped — and the tool's problem format can't represent kernel fusion at all, a real scope limit deferred to RTL.
- Validated the hypothesis against real Gemmini RTL on Verilator: confirmed a native hardware softmax unit whose running-max/running-sum/max-subtracted-exp structure matches the online-softmax mechanism independently derived by hand, and confirmed (via Gemmini's actual transposer hardware) the assumption that one physical array can serve both attention matmuls. Also found that a "weight-stationary-only" config restricts *behavior* via a compile-time control constant on an otherwise-generic PE, not by generating structurally smaller hardware — a real, non-obvious revision to the original hardware-cost framing.

**Decode:**
- **GQA flips from a secondary to a first-order lever.** Fused prefill gets *zero* throughput benefit from GQA (compute-bound, FLOPs-driven) — decode, where `seq_len_q` collapses to 1, sees GQA's byte savings pass through almost fully (**~8× AI improvement, 2 → 16 FLOPs/byte**), because the Q/output terms that capped prefill's win to 1.78× collapse to near-zero. Same technique, opposite payoff, purely a function of regime.
- **Hand-found a hardware-breaking issue before touching any tool** (see headline above): a systolic array's fill/drain amortization needs a deep temporal stream that decode structurally can't provide — forcing a pivot to a SIMD/vector compute primitive, confirmed entirely unbuildable in Gemmini's actual generator (no vector/SIMD compute path exists — only the systolic array).
- Every hardware difference between the prefill and decode hypotheses — systolic vs. SIMD, `d_head`- vs. `seq_len_kv`-parallelized, forced-sequential vs. fully concurrent GQA reuse, power-of-2-capped vs. strip-mined SRAM sizing — traces to the **same single root cause**: `seq_len_q` collapsing from 8192 to 1, not a list of unrelated design choices. Matches the roofline numbers precisely: prefill's hardware chases throughput (~17× above ridge); decode's avoids wasting silicon on compute capability the workload structurally can't use (30–240× below ridge).
- **Numerics is a decisive lever for the sibling MoE project below, but structurally powerless for decode attention** (see headline above) — the crossover to flip decode's regime would need ~0.27 bits/element, not a realizable format. The generalizable finding: whether a numerics lever can flip a regime is a question of **how large the margin is that it has to close**, not whether the lever exists in principle.
- Phase 2c (Timeloop) and 2d (Gemmini RTL) were deliberately skipped for decode — 2d because Gemmini has no SIMD compute path to validate against at all (a harder gap than prefill's fusion-modeling limit), 2c because its real payoff in Phase 1 was tool-methodology lessons already banked and actively applied here without re-running the tool. Phase 4 (real-hardware check) skipped as already substantively covered by the real RTL work above. Full reasoning in `decode_notes.md`.

**Roofline — MHA vs. GQA, prefill** (batch=32, seq_len=8192, int8):

| | FLOPs | Bytes (fused) | Bytes (unfused) | AI (fused) | AI (unfused) |
|---|---|---|---|---|---|
| MHA | 2⁴⁶ | 8 GiB | ≈520 GiB | 8,192 | ≈126 |
| GQA | 2⁴⁶ | 4.5 GiB | ≈516.5 GiB | ≈14,564 | ≈126.9 |

**Roofline — MHA vs. GQA, decode** (batch=32, seq_len_kv=8192, seq_len_q=1, int8):

| | FLOPs | Bytes | AI | vs. ridge (480.5) |
|---|---|---|---|---|
| MHA | 2³³ | ≈4.0005 GiB | ≈2.0 | ~240× below |
| GQA | 2³³ | ≈0.5005 GiB | ≈15.98 | ~30× below |

| Phase | Focus | Status |
|---|---|---|
| 1a | Prefill roofline: FLOPs, bytes, AI, ridge point | ✅ done |
| 1b | Prefill hardware hypothesis: PE array, dataflow, scratchpad/accumulator | ✅ done |
| 1c | Sweep the same design space in Timeloop, compare to 1b | ✅ done |
| 1d | Configure Gemmini, read generated RTL, validate in Verilator | ✅ done (deliberately scoped: RTL literacy + open-question resolution, not a full custom kernel/quantitative cycle comparison — see writeup for why) |
| 2a | Decode roofline: FLOPs, bytes, AI, ridge point | ✅ done |
| 2b | Decode hardware hypothesis + explicit prefill-vs-decode comparison | ✅ done |
| 2c/2d | Timeloop sweep / Gemmini RTL validation for decode | ⏭️ deliberately skipped — Gemmini has no SIMD compute path at all (verified against source); 2c's marginal value judged low given tool-methodology lessons already banked in 1c |
| 3 | Numerics: is KV-cache quantization a bigger lever than GQA? | ✅ done (reframed from a generic precision-throughput comparison) — no, decode's margin is too large for any realizable format to close |
| 4 | Real-hardware sanity check | ⏭️ deliberately skipped — already substantively covered by 1d's real RTL work and prior independent work |

Full writeups: [`prefill_notes.md`](prefill_notes.md) (all four sub-phases, cross-phase synthesis, open threads, toolchain notes) and [`decode_notes.md`](decode_notes.md) (2a/2b, the explicit cross-phase hardware comparison, Phase 3 numerics, and full scope reasoning for 2c/2d/4).

---

## MoE Routing System Codesign → [`moe-routing-notes.md`](moe-routing-notes.md)

**Stack:** ASTRA-sim (setup pending) · **Workload:** DeepSeek-V2 MoE (236B total / 21B activated) — 160 routed + 2 shared experts, top_k=6 capped at M≤3 devices, MLA attention, 8-way expert-parallel deployment on TPU 8i (FP4, Boardfly interconnect)

Extends the same loop one level up the stack: from single-accelerator microarchitecture to multi-chip system architecture, using MoE token routing as the workload — data-dependent destinations, not a fixed shape, so load imbalance (not just AI) has to be modeled.

- Derived dispatch+combine communication volume (**105 MiB/layer**, system-wide, one decode step) against DeepSeek-V2's actual device-limited routing mechanism (verified against the primary paper source), rather than the textbook `top_k × tokens / num_experts` formula, which overcounts fan-out.
- Compute-to-comms ratio: **≈28,087 FLOPs/byte** vs. a TPU 8i ICI ridge point of **≈4,208** — decisively compute-bound, **6.7× margin**, in the ideal/uniform-routing case.
- Modeled load imbalance against two real, cited sources (Gini≈0.70 across DeepSeek-V3/Qwen3-MoE/Mixtral; 70% GPU stall time measured on real Mixtral-8×7B serving) and found a **hard, imbalance-proof floor of ≈21,065 FLOPs/byte** — no routing skew, however extreme, flips this workload comms-bound on this hardware.
- The lever that actually matters isn't imbalance — it's **numerics**: the floor scales inversely with dispatch precision, and the exact crossover to comms-bound is **≈2.5 bytes/element** (between FP8 and BF16). This is the realistic-crossover case that decode attention's own numerics analysis (above) contrasts directly against — same lever, opposite verdict, purely a function of starting margin size.
- Under DeepSeek-V2's real device-level token-dropping policy (**CF = 1.0**, the strictest possible setting), a device at realistic imbalance severity would reject **~70% of its excess demand** — a concrete, quantified quality/throughput tradeoff, not a hand-wave.

| Phase | Focus | Status |
|---|---|---|
| 1a–1c | Workload shape, uniform + imbalanced communication volume | ✅ done |
| 2 | System architecture hypothesis: topology, bandwidth, buffering | ✅ done |
| 3 | Validate the hypothesis in ASTRA-sim | ✅ done — topology confirmed; bandwidth/buffering found untestable in the analytical backend (a real tool-ceiling finding, not a gap in the reasoning) |
| 4 | Synthesis: on-chip SRAM (attention) vs. interconnect (MoE) budget tradeoff | ✅ done |

Full writeup (workload, hypothesis, ASTRA-sim results, cross-project
synthesis, key takeaways): [`moe-routing-notes.md`](moe-routing-notes.md).

---

*Self-directed projects, not course assignments — this README reflects exactly where each stands today, not a finished scope.*
