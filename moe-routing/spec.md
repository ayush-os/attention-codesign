# Project Spec: Workload → System Codesign for MoE Token Routing (Multi-Chip)

**Goal:** Extend the "workload → hardware" reasoning loop from the attention project up one level — from single-accelerator microarchitecture (array shape, dataflow, scratchpad/accumulator sizing) to system-level architecture (interconnect topology, bandwidth allocation, load balancing) — using MoE expert routing as the workload, since routing is inherently a multi-chip communication problem, not a single-device one.

**Prerequisite mindset shift:** in the attention project, the workload's shape was fixed and known ahead of time (batch, seq_len, heads — all static). MoE routing is **data-dependent**: which expert a token goes to depends on the token itself, so load across experts/chips is not uniform, and worst-case behavior matters as much as average-case. Expect more of this project to be about reasoning under uncertainty (imbalance, tail behavior) than the attention project was.

**Legend:** 🔧 = boilerplate/setup, given directly. 🧠 = your job — research, derive, decide, and be prepared to defend the choice.

---

## Phase 0: Setup (🔧 boilerplate)

- Install [ASTRA-sim](https://astra-sim.github.io/) (Georgia Tech / Meta / Intel) — a network-and-compute co-simulator built for exactly this class of problem (collective communication across accelerator topologies). This replaces Timeloop/Gemmini's role from the attention project; those tools model a single accelerator's internal datapath, not a multi-chip network, so they're the wrong instrument here.
- Skim ASTRA-sim's docs/paper enough to know what it takes as input: a topology description (ring, mesh, switch-based, etc.), per-link bandwidth/latency, and a workload trace (compute + communication events).
- Optionally, keep a lightweight hand-rolled network model (a simple Python script computing time = max(compute, communication) per stage) as a sanity-check tool you fully control, the way you cross-checked Timeloop against hand analysis before.

**Checkpoint:** you can run ASTRA-sim's example workload (e.g. an all-reduce on a stock topology) before modifying anything.

---

## Phase 1: Characterize the workload (🧠)

### 1a. Pick a concrete MoE shape
Pick real numbers, sourced the same way you sourced Llama 3-70B for attention (scaling book, published model reports, etc.) — e.g. a Mixtral- or DeepSeek-style layer: num_experts, top_k, hidden_dim, num_chips (experts sharded how many ways), batch/tokens-per-step. State your source.

### 1b. Derive communication volume — the uniform-routing case first
As a control case (mirroring how you did naive MHA before GQA), assume **perfect load balance**: every expert receives exactly `(top_k × tokens) / num_experts` tokens. Derive:
- Bytes moved per token in the dispatch (token → assigned expert's chip) and combine (expert output → back to originating chip) phases.
- Total communication volume for one MoE layer's forward pass under this assumption.
- Compare this to the FLOPs done by the experts themselves — what's the compute-to-communication ratio, and does it suggest this workload is comms-bound or compute-bound *in the ideal case*?

### 1c. Derive communication volume under load imbalance — the real case
🧠 This is the genuinely new problem attention didn't have. Research and decide:
- What's a realistic (not worst-case-pathological, not perfectly-uniform) model for expert load imbalance? Look into what's actually published about routing imbalance in real MoE systems (capacity factor, token dropping, auxiliary load-balancing losses) — this is a real, actively-discussed problem in the literature, not something to invent from scratch.
- Given a chosen imbalance model, what happens to your Phase 1b numbers? Which chip becomes the bottleneck, and by how much versus the uniform case?
- State this as a range (best case / expected case / worst case), the same way you carried fused/unfused as two bounds for attention rather than picking one number.

**Deliverable for Phase 1:** hand-derived communication volume (uniform + imbalanced cases), compute-to-communication ratio, and a first-pass prediction of whether/when this workload is network-bound.

---

## Phase 2: Predict the ideal system architecture (🧠)

Mirroring Phase 1b from the attention project, but one level up the stack:

- **Topology**: given your Phase 1 communication pattern (all-to-all-ish, data-dependent destination), what interconnect topology would you hypothesize is well-suited — and why? Research real options (ring, fat-tree, switch-based, dragonfly) and reason about which properties of *this specific* communication pattern (not communication patterns in general) matter most: is it more sensitive to bisection bandwidth, to latency, to worst-case link contention under imbalance?
- **Bandwidth allocation**: given a fixed total interconnect budget, would you provision uniformly across links, or does your Phase 1c imbalance analysis argue for something else (e.g. extra headroom on links serving popular experts)? This is a real, non-obvious design question — reason it through rather than defaulting to "uniform is simplest."
- **Buffering / SRAM implications**: tokens in flight during dispatch need to sit somewhere. Given your Phase 1 volumes, roughly how much on-chip buffering would a chip need to avoid stalling under your imbalance model? (Same style of scratchpad-sizing reasoning as the attention project's §3, just for network buffers instead of matmul operands.)

Write this down as a defensible hypothesis, same as before — not required to be correct yet.

---

## Phase 3: Test the hypothesis in ASTRA-sim (🔧 tool use, 🧠 interpretation)

- Configure ASTRA-sim with your Phase 2 hypothesis (topology, bandwidth, your workload's compute+communication trace).
- Sweep topology and bandwidth-allocation variants around your hypothesis.
- Compare the tool's near-optimal configuration against your Phase 2 prediction. Where do they agree? Where don't they, and why — same gap-hunting standard as the attention project. Does the imbalance model you chose in 1c actually show up as the bottleneck ASTRA-sim's simulation predicts, or does something else dominate (e.g. per-hop latency you underweighted)?

**Deliverable for Phase 3:** predicted vs. simulated comparison, with every meaningful gap explained mechanistically, not just noted.

---

## Phase 4: Connect back to the attention project (🧠, the differentiating step)

This is the step that makes the two projects one coherent body of work instead of two disconnected exercises:

- In a real deployment, attention (Phase 1 project) happens *on* individual chips, and MoE routing (this project) happens *across* them, in the same forward pass, competing for the same power/area/bandwidth budget. Reason about the interaction: does provisioning more interconnect bandwidth for MoE (this project) come at the expense of on-chip SRAM for attention's scratchpad/accumulator (previous project)? Is there a workload-dependent argument for how a chip's design should trade off "more on-chip memory for attention" against "more interconnect bandwidth for MoE dispatch"?
- This doesn't need a fully rigorous answer — a well-reasoned qualitative argument, grounded in the actual numbers from both projects, is the goal. This is exactly the kind of cross-cutting tradeoff a real chip architecture team has to make and is a strong differentiator for a writeup/interview story.

---

## Final Deliverable: Writeup

Structure:
1. **Workload**: MoE shape chosen, source, uniform vs. imbalanced communication derivation.
2. **System hypothesis**: topology, bandwidth allocation, buffering — predicted and why.
3. **ASTRA-sim results**: comparison to hypothesis, gaps explained.
4. **Cross-project synthesis**: how attention's on-chip needs and MoE's off-chip needs interact/compete for the same budget.
5. **Reflection**: what surprised you, what a next extension would be (e.g. modeling MoE + attention together in one full-system simulation, or pipeline-parallelism-driven communication instead of expert-routing-driven).

---

## Rough Timeline

| Phase | Focus |
|---|---|
| 0 | ASTRA-sim setup |
| 1 | Workload characterization: uniform + imbalanced communication volume |
| 2 | Hand-derived system hypothesis: topology, bandwidth, buffering |
| 3 | ASTRA-sim sweep + gap analysis |
| 4 | Cross-project synthesis + writeup |

Exact per-phase timing is deliberately left open — you moved faster than a naive estimate on the attention project, and Phase 1 here (imbalance modeling) is the phase most likely to take longer than it looks, since it requires real research into published load-balancing approaches rather than pure derivation.

## Fallback / Minimum Viable Version
Phases 1-3 on the **uniform-routing case only** (skip imbalance modeling entirely) still produces a complete, defensible artifact — a full workload→topology→validated-in-simulation loop, just without the data-dependent-routing complexity. Imbalance modeling (1c) and the cross-project synthesis (Phase 4) are the highest-value additions if time allows, in that order.
