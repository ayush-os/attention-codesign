# Workload → Silicon: Hardware & System Codesign for LLM Inference

Two self-directed codesign studies — single-chip attention microarchitecture, then multi-chip MoE routing — every prediction hand-derived and logged *before* being checked against a tool or the literature.

**Stack:** Timeloop/Accelergy, Gemmini (Chipyard), Verilator, ASTRA-sim · **Workloads:** Llama 3-70B prefill attention (GQA) · DeepSeek-V2 MoE routing (multi-chip, 8-way EP)

- **The compute/memory-bound regime is a design decision, not a workload property**: the same attention workload hits **AI = 8,192 FLOPs/byte** (compute-bound, fused) or **AI ≈ 126** (memory-bound, unfused) against a TPU v5e ridge point of **≈480.5** — the fusion decision alone swings the regime **65×**.
- **Independently found a GQA-breaking bug by hand**, before touching Timeloop or Gemmini: a naive accumulator-driven `tile_q=32` forces Q-tile-outer loop ordering, causing **~256× K/V re-fetching from HBM** — breaking GQA's textbook "fetch once" reuse claim.
- **Derived DeepSeek-V2's real dispatch/combine traffic** against the paper's actual device-limited routing mechanism (not the textbook formula) and proved the workload sits on a **hard, imbalance-proof compute-bound floor** (~21,065 FLOPs/byte, ~5× the TPU 8i ICI ridge point) — no routing skew, however severe, can flip it comms-bound. Only a numerics choice (dispatch precision) can.

---

## Methodology (shared across both projects)

Every phase follows the same loop: **hand-derive a prediction → validate against a tool or real literature → explain every gap mechanistically.** Applied first at the single-accelerator level (PE array, dataflow, scratchpad), then one level up at the system level (interconnect topology, bandwidth, buffering).

---

## Attention Hardware Codesign → [`attention/`](attention/)

**Stack:** Timeloop/Accelergy, Gemmini (Chipyard), Verilator · **Workload:** Llama 3-70B prefill attention (GQA), from *How to Scale Your Model*'s TPU-serving numbers

- Proved the compute/memory-bound regime is a **design decision, not a workload property** (see above) — a 65× AI swing from fusion alone, at a fixed ridge point.
- Showed GQA's 8× KV-cache reduction is **not** an 8× bandwidth win: total compulsory bytes only drop **1.78×** (Amdahl's Law — Q/output activity floors the total), and in the fused/compute-bound regime GQA doesn't reduce execution time *at all*, since time is set by FLOPs, which GQA leaves untouched.
- Derived a full hardware hypothesis from first principles (**128×128** systolic array, K/V-stationary weight-stationary dataflow, `tile_k=1024`, `tile_q=32`) and, stress-testing it by hand, found the `tile_q=32` re-fetch tension above — a sharpened, falsifiable prediction for the Timeloop sweep, not a fixed conclusion.
- Swept the hypothesis in Timeloop and found the ridge point itself doesn't transfer across chips: a small single-array model has **6× lower peak compute** than TPU v5e at the same real HBM bandwidth, so its ridge point is **≈80, not 480.5** — flipping what "unfused" predicts for *this* hardware specifically. Also caught the mapper tool's own blind spot: its search is deterministic (confirmed via a byte-for-byte-identical rerun), so a discovered "optimum" can just be a local one a shallow search never escaped — and the tool's problem format can't represent kernel fusion at all, a real scope limit now deferred to RTL rather than hidden.

**Roofline — MHA vs. GQA** (batch=32, seq_len=8192, int8):

| | FLOPs | Bytes (fused) | Bytes (unfused) | AI (fused) | AI (unfused) |
|---|---|---|---|---|---|
| MHA | 2⁴⁶ | 8 GiB | ≈520 GiB | 8,192 | ≈126 |
| GQA | 2⁴⁶ | 4.5 GiB | ≈516.5 GiB | ≈14,564 | ≈126.9 |

| Phase | Focus | Status |
|---|---|---|
| 1a | Roofline: FLOPs, bytes, AI, ridge point | ✅ done |
| 1b | Hardware hypothesis: PE array, dataflow, scratchpad/accumulator | ✅ done |
| 1c | Sweep the same design space in Timeloop, compare to 1b | ✅ done |
| 1d | Configure Gemmini, read generated RTL, validate in Verilator | ⏳ next |
| 2 | Repeat 1a→1d for the decode regime (memory-bound by construction) | ⏳ not started |

Full derivation trail (every hypothesis, correction, and dead end): [`attention/notes.md`](attention/notes.md).

---

## MoE Routing System Codesign → [`moe-routing/`](moe-routing/)

**Stack:** ASTRA-sim (setup pending) · **Workload:** DeepSeek-V2 MoE (236B total / 21B activated) — 160 routed + 2 shared experts, top_k=6 capped at M≤3 devices, MLA attention, 8-way expert-parallel deployment on TPU 8i (FP4, Boardfly interconnect)

Extends the same loop one level up the stack: from single-accelerator microarchitecture to multi-chip system architecture, using MoE token routing as the workload — data-dependent destinations, not a fixed shape, so load imbalance (not just AI) has to be modeled.

- Derived dispatch+combine communication volume (**105 MiB/layer**, system-wide, one decode step) against DeepSeek-V2's actual device-limited routing mechanism (verified against the primary paper source), rather than the textbook `top_k × tokens / num_experts` formula, which overcounts fan-out.
- Compute-to-comms ratio: **≈28,087 FLOPs/byte** vs. a TPU 8i ICI ridge point of **≈4,208** — decisively compute-bound, **6.7× margin**, in the ideal/uniform-routing case.
- Modeled load imbalance against two real, cited sources (Gini≈0.70 across DeepSeek-V3/Qwen3-MoE/Mixtral; 70% GPU stall time measured on real Mixtral-8×7B serving) and found a **hard, imbalance-proof floor of ≈21,065 FLOPs/byte** — no routing skew, however extreme, flips this workload comms-bound on this hardware.
- The lever that actually matters isn't imbalance — it's **numerics**: the floor scales inversely with dispatch precision, and the exact crossover to comms-bound is **≈2.5 bytes/element** (between FP8 and BF16).
- Under DeepSeek-V2's real device-level token-dropping policy (**CF = 1.0**, the strictest possible setting), a device at realistic imbalance severity would reject **~70% of its excess demand** — a concrete, quantified quality/throughput tradeoff, not a hand-wave.

| Phase | Focus | Status |
|---|---|---|
| 1a | Workload shape: DeepSeek-V2, MLA, device-limited routing | ✅ done |
| 1b | Communication volume, uniform routing (ideal case) | ✅ done |
| 1c | Communication volume under load imbalance | ✅ done |
| 2 | System architecture hypothesis: topology, bandwidth, buffering | ⏳ next |
| 3 | Validate the hypothesis in ASTRA-sim | ⏳ not started |
| 4 | Synthesis: on-chip SRAM (attention) vs. interconnect (MoE) budget tradeoff | ⏳ not started |

Full derivation trail: [`moe-routing/notes.md`](moe-routing/notes.md).

---

*Self-directed projects, not course assignments — this README reflects exactly where each stands today, not a finished scope.*
