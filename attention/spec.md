# Project Spec: Workload → Hardware Codesign for Attention (Prefill & Decode)

**Goal:** Build calibrated intuition for going from *workload characteristics* to *hardware design decisions*, then validate that intuition in RTL — closing the loop that separates a "codesigner" from a "kernel engineer." Contrast two regimes (compute-bound prefill vs. memory-bound decode) so the intuition generalizes rather than overfitting to one workload.

**Timeframe:** ~2-3 weeks core (Phases 1-3), + optional Phase 4 if time allows.

**Legend:** 🔧 = boilerplate/setup, given directly. 🧠 = your job — research, derive, decide, and be prepared to defend the choice.

---

## Phase 0: Setup (🔧 boilerplate, budget ≤2 days)

- Install [Timeloop + Accelergy](https://timeloop.csail.mit.edu/) (MIT). Use their tutorial infra/Docker image — don't hand-build the environment from scratch, that's not the learning-bearing part.
- Install [Chipyard](https://chipyard.readthedocs.io/) with [Gemmini](https://github.com/ucb-bar/gemmini) integrated. Get the default example (matmul) running through Verilator so you know the toolchain works before you start modifying anything.
- Skim (don't deep-read yet) the Gemmini paper and the Timeloop/Accelergy papers, just enough to know what knobs exist: PE array dimensions, dataflow (weight/output/input-stationary), scratchpad size, accumulator size, DMA/bandwidth parameters.

**Checkpoint:** you can run the stock example end-to-end in both tools before moving on.

---

## Phase 1: Prefill attention (compute-leaning regime)

### 1a. Characterize the workload (🧠)
Pick a concrete, fixed shape (batch size, sequence length, head dim, num heads — pick something realistic, e.g. inspired by a model size you know from CS336/scaling book). You derive:
- FLOPs and bytes-moved for the prefill attention computation (QK^T, softmax, ·V) at this shape.
- Arithmetic intensity (FLOPs/byte).
- Where that lands relative to a rough "ridge point" for a plausible accelerator — i.e., do you expect this to be compute-bound or memory-bound, and *why*, before touching any tool.

Write this down before running anything. This prediction is what you're testing.

### 1b. Predict the ideal hardware shape (🧠)
Using your Phase 1a numbers, reason about:
- What PE array shape would keep utilization high for this workload's matmul dimensions (not too big that you underfill it, not too small that you're bound elsewhere)?
- Which dataflow (weight-stationary / output-stationary / input-stationary) is most natural for this workload's reuse pattern, and why?
- Roughly how big should the scratchpad be to hold what needs reuse without spilling?

You don't need to be right — you need a defensible hypothesis to test against Phase 1c.

### 1c. Search the space in Timeloop/Accelergy (🔧 tool use, 🧠 interpretation)
- Sweep array shape, dataflow, and memory sizing in Timeloop for your fixed workload shape.
- Find what the tool considers near-optimal.
- Compare against your Phase 1b hypothesis. Where do they agree? Where don't they, and what did your hand analysis miss (this is the important part — don't skip explaining the mismatch)?

### 1d. Build it in Gemmini and validate (🔧 config, 🧠 RTL reading + debugging)
- Configure Gemmini as close as the generator allows to your Timeloop-derived "winning" config.
- **Actually read the generated RTL for your config** — trace how your array-size/dataflow choices show up as real datapath structure. (You have the EE108/EE180 background for this — use it.)
- Run your attention kernel (or a representative slice of it) through Verilator, get real cycle counts / utilization.
- Compare against Timeloop's prediction. Explain every meaningful gap mechanistically (fixed overheads Timeloop doesn't model? DMA granularity? instruction issue effects?). This gap-hunting is the highest-value activity in the whole project — don't rush past it.

**Deliverable for Phase 1:** a short internal writeup (rough notes are fine at this stage) with your prediction, the Timeloop result, the Gemmini/RTL result, and your explanation of every divergence.

---

## Phase 2: Decode attention / KV-cache-bound inference (memory-leaning regime)

Repeat the *exact same loop* (1a → 1d) for decode-phase attention: small batch, single new token, dominated by reading the growing KV cache rather than large matmuls.

**🧠 Before you start:** think through, from first principles, why this regime should push your hardware conclusions in a different direction than prefill did — larger effective memory bandwidth need relative to compute, different reuse pattern, possibly a different "ridge point" crossover. Write your prediction down before repeating 1b-1d for this workload. Don't just copy Phase 1's config and assume it transfers — that assumption is exactly what you're testing.

**Deliverable for Phase 2:** same structure as Phase 1 — prediction, Timeloop result, Gemmini/RTL result, gap explanation — plus an explicit comparison: how did your "ideal" array shape, dataflow, and scratchpad sizing differ between prefill and decode, and does that match what you'd expect from the two workloads' arithmetic intensity?

---

## Phase 3: Numerics extension (🧠, connects to your Rivos FP8/MXFP8 background)

- Using your best config(s) from Phases 1-2, compare precision modes Gemmini supports (e.g., int8 vs. higher precision) on throughput/cycles.
- Reason about (you won't get exact numbers without real synthesis, and that's fine — reason qualitatively with rough estimates) what the *hardware cost* of supporting lower precision looks like: extra multiplier modes, dequant/scaling logic, wider accumulate paths. What's the throughput gain buying you, and what would you guess it costs?
- Tie this explicitly to what you already know from Rivos — does your production kernel-porting experience with FP8/MXFP8 change how you think about this hardware tradeoff?

---

## Phase 4 (optional, only if time remains): Real-hardware sanity check

Take your core conclusion — e.g., "decode should cross into memory-bound territory at roughly X arithmetic intensity, prefill stays compute-bound past that point" — and spot-check it against real behavior on a rented H100/B200 or TPU/Trainium if accessible. This isn't a full re-analysis — just enough to see whether your simulated intuition survives contact with real silicon. Note where it does and doesn't; both outcomes are useful for the writeup.

---

## Final Deliverable: Writeup

Structure (this part is 🔧, just organize what you've already produced):
1. **Setup** — workload shapes chosen, why.
2. **Prefill**: hand roofline → Timeloop-predicted config → Gemmini/RTL-validated result → gap analysis.
3. **Decode**: same structure.
4. **Comparison**: how and why the "ideal" hardware differs between the two regimes.
5. **Numerics**: precision/hardware-cost tradeoff reasoning.
6. **(Optional) Real-hardware check.**
7. **Reflection**: what surprised you, what you'd want to explore next (e.g., MoE routing, multi-chip sharding effects on hardware design).

This writeup — two workloads, opposite bottleneck profiles, a full hand→simulated→RTL→gap-explained loop, plus a numerics extension grounded in real prior experience — is the artifact. It's what you'd walk an interviewer through at MatX, Anthropic, or OpenAI's hardware team.

---

## Rough Timeline

| Week | Focus |
|---|---|
| 1 | Phase 0 setup + Phase 1 (prefill): hand analysis → Timeloop → Gemmini/RTL |
| 2 | Phase 2 (decode): full loop repeated + comparison |
| 3 | Phase 3 (numerics) + Phase 4 (optional real-hw check) + writeup |

If you're moving faster than expected, spend the slack going deeper on gap explanations (Phase 1d/2d) rather than adding a third workload — the depth of "why did theory diverge from reality" is worth more than breadth.

## Fallback / Minimum Viable Version
If time gets eaten (MTIA work, life, etc.): Phase 1 alone (prefill, full loop, written up) is still a complete, defensible artifact. Decode, numerics, and real-hardware validation are additive, not required.
