# Project Spec: Disaggregated Serving + Weight/KV-Cache Placement

**Goal:** extend from "which chips run which phase" (disaggregation) down to "within that architecture, what actually lives in SRAM vs. HBM vs. gets fetched remotely, and how does data move as prefill hands off to decode." This is the mechanism underneath your MoE and attention projects' findings, applied at the memory-hierarchy level.

**Legend:** 🔧 = boilerplate/setup. 🧠 = your job.

---

## Phase 0: Setup (🔧, with one 🧠 twist)

Unlike Timeloop/Gemmini (attention) and ASTRA-sim (MoE), there isn't a clean off-the-shelf tool purpose-built for "disaggregated serving + hierarchical KV placement" tradeoff exploration.
- 🔧 Read the real systems this space is built on: **vLLM's PagedAttention** implementation (real, inspectable code), the **DistServe** paper (prefill/decode disaggregation), and the **Mooncake** paper (KV-cache-centric disaggregated architecture, published by Moonshot AI — directly relevant given it's literally about this problem at production scale).
- 🧠 **You'll likely need to hand-build a lightweight analytical/simulation model** rather than rely on a single existing tool — this mirrors what you did as a sanity-check alongside Timeloop, just promoted to the primary instrument here. Decide what it needs to model: memory capacities/bandwidths at each level (SRAM, HBM, interconnect), request arrival pattern, KV cache growth per request, and eviction/transfer costs. This is a real, valuable design decision in itself — don't skip past it.

**Checkpoint:** you have real reference numbers from DistServe/Mooncake (their reported latency/throughput tradeoffs) to sanity-check your own model against later.

---

## Phase 1: Characterize the memory hierarchy problem (🧠)

- Using your attention project's own Llama-3-70B numbers (you already have weight sizes, KV-cache-per-token bytes from Phase 1a), derive: how many tokens of KV cache fit in one chip's SRAM? HBM? At what context length does a single request's KV cache alone exceed on-chip capacity?
- Derive the **growth curve**: KV cache size as a function of context length, batch size, num_kv_heads (bring back your GQA numbers directly — this is exactly the lever you already quantified).
- State the core tension explicitly: weights are static/huge/fixed; KV cache is dynamic/growing/per-request. What does that imply about what *should* stay SRAM-resident (candidates: hot expert weights from MoE, attention weights) vs. what's fundamentally HBM/remote-bound (KV cache at scale)?

---

## Phase 2: Disaggregation hypothesis (🧠)

- Using your attention project's prefill (compute-bound) vs. decode (memory-bound) roofline findings directly: hypothesize a **chip ratio** (how many prefill chips per decode chip) that balances throughput between the two pools, given your workload's request pattern (prompt length vs. generation length distribution — pick something realistic and state your source).
- Hypothesize the **handoff mechanism**: when a request finishes prefill, its KV cache must reach the decode chip. What's the transfer cost (bytes = your Phase 1a KV cache size), and over what interconnect (reuse your MoE project's topology reasoning — is this the same network fabric, or does it warrant a dedicated one)?

---

## Phase 3: KV/weight placement policy (🧠)

- Hypothesize an eviction/placement policy: given limited SRAM/HBM, what stays resident and what gets evicted/offloaded as context grows or batch pressure increases? Research real approaches (PagedAttention's paging, KV cache quantization, attention sink / sliding window approaches, offload-to-CPU) rather than inventing one from scratch — this is an active, well-published area.
- For MoE-relevant weights specifically: connect back to your MoE imbalance findings — should "hot" experts (per your Phase 1c imbalance model) get preferential SRAM residency over cold ones? This is a genuinely new question your MoE project set up but didn't answer.

---

## Phase 4: Validate (🔧 build, 🧠 interpret)

- Implement your Phase 0 model with your Phase 2/3 hypotheses plugged in.
- Compare your model's predicted latency/throughput tradeoff against DistServe's/Mooncake's published results (order-of-magnitude sanity check, not exact reproduction — you don't have their exact workload/hardware).
- Explain gaps the same way you always have: mechanistic, not hand-wavy.

---

## Phase 5: Full synthesis across all three projects (🧠, the capstone)

Write the connective argument across attention → MoE → this project: given a fixed chip/power/area/bandwidth budget for a whole rack, how should it be split across (a) on-chip SRAM for attention scratchpad/accumulator, (b) on-chip SRAM for hot MoE expert weights and KV cache, (c) interconnect bandwidth for MoE dispatch, (d) interconnect bandwidth for prefill→decode KV handoff. This is the actual rack-scale codesign story, built from three pieces of real derived evidence instead of speculation.

---

## Rough Timeline
Same open-ended structure as before — Phase 0-1 setup/characterization, Phase 2-3 the hypothesis-building (likely the longest, most research-heavy phases here), Phase 4 validation, Phase 5 the synthesis writeup tying all three projects together.

## Fallback
Phases 1-3 (hand analysis + hypothesis, no simulation) still stands alone as a complete artifact if time runs short — Phase 4 (validation) and Phase 5 (cross-project synthesis) are the highest-value additions if time allows.

---

## Note on scope (why this stayed a focused third project, not a monolithic e2e system)

The thing that made the attention and MoE projects valuable wasn't breadth — it was depth per decision (e.g. the K/V-reuse/accumulator-capacity tension found by walking the full loop nest by hand). A single project trying to simultaneously handle sharding + disaggregation + MoE + KV placement would force shallow treatment of each piece just to stay tractable, trading a real mechanistic finding for a system diagram. Three focused, sequential projects — each with an explicit synthesis connecting it to what came before — build the same end-to-end picture of rack-scale inference without ever going shallow.