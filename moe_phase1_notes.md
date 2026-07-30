# MoE Routing Project — Notes (Phase 0 / Phase 1a)

Companion to `moe_routing_project_spec.md`. Mirrors the collaboration mode and
logging style of `phase1_notes.md`/`handoff.md` from the attention codesign
project — this is the second project in that same self-directed arc.

---

## Collaboration mode (same as the attention project — read first)

This is a **self-directed learning project**. Job is to check reasoning, ask
questions that expose gaps, flag missing considerations, help structure/log —
**not** solve steps or hand over derivations the user hasn't produced
themselves.

Exceptions, all explicitly confirmed by the user in this project too:
- **Pure arithmetic plug-in after the user has established the formula** —
  user has explicitly said this is fine to hand over ("I've done the *How to
  Scale Your Model* book e2e... it's a waste of my time to do this math").
  Still don't derive the formula itself for them.
- **Factual/reference lookups** (real hardware specs, paper details, "does
  chip X support format Y") — verify via search, don't trust memory. Several
  numbers in this doc (TPU 8i specs, MLA equations) were pulled this way and
  are cited.
- Chip/precision *selection* (as opposed to conceptual derivation) was
  explicitly marked by the user as a "just recommend, don't Socratic-method
  me" category — direct recommendations were given for those, with reasoning,
  not posed as questions.
- Batch-size arithmetic and the final weights/KV-cache computation were
  explicitly delegated to be computed directly, once the user had stated the
  formulas.

The user works well with concrete, quantified pushback (e.g., "that's 8x
redundant KV storage — is that intentional?") rather than abstract objections.

---

## Phase 0 status

- **ASTRA-sim setup, not yet executed** (decided, not done). Plan: attempt
  local build on Mac first (M3 Max, 36GB RAM, 14 cores) via Docker — verified
  the repo's Dockerfile is `FROM ubuntu:22.04` with no architecture pinning,
  so Docker Desktop should pull/build natively on arm64 rather than emulate,
  unlike Chipyard. The one unverified piece is the optional NS-3 network
  backend (heavier C++ build; analytical backend is the lighter, default
  path and is what the Phase 0 checkpoint — running the stock all-reduce
  example — actually needs).
- **Farmshare (Stanford HPC) kept as fallback, not default venue** — for (a)
  if NS-3 specifically fails to build under arm64/Docker, or (b) if Phase 3's
  topology/bandwidth sweeps turn into long batch jobs worth running headless.
  Reasons *against* defaulting to Farmshare: Docker daemon access on shared
  academic clusters is unconfirmed (may require Singularity/Apptainer
  instead), plus shared-resource contention and ssh/iteration friction that
  a fully local Mac workflow doesn't have.
- Confirmed while deciding this: local 1c (Timeloop/Accelergy) work for the
  *attention* project was concurrently saturating ~6 of 10 P-cores at the
  time — informed the "don't build ASTRA-sim locally in parallel with an
  active Timeloop sweep" call, though this doesn't apply once 1c finishes.

---

## Phase 1a: Workload shape — DeepSeek-V2

**Source:** DeepSeek-V2 paper (arXiv 2405.04434), §1 and §3.1 (model
hyperparameters), plus §2.1 (MLA architecture, equations 1–19).

### Locked-in shape parameters

| Parameter | Value | Source |
|---|---|---|
| Routed experts | 160 | paper §3.1 |
| Shared experts | 2 | paper §3.1 |
| top_k (routed) | 6 | paper §3.1 |
| Device-limited routing cap (M) | ≤3 of the 8 devices per token | paper, device-limited routing section |
| Expert FFN intermediate dim (d_ff) | 1536 | paper §3.1 |
| d_model | 5120 | paper §3.1 |
| Transformer layers (L) | 60 (**note:** layer 1 is a dense FFN, not MoE — not yet incorporated into the arithmetic below, flagged as a small known gap) | paper §3.1 |
| Total / activated params | 236B / 21B | paper §3.1 |
| num_chips (D), per-layer EP group | 8 — "routed experts uniformly deployed on 8 devices" | paper, expert-parallel deployment section |
| n_heads (n_h) | 128 | paper §3.1 |
| d_head (d_h) | 128 | paper §3.1 |
| KV compression dim (d_c) | 512 | paper §3.1 |
| Query compression dim (d_c') | 1536 | paper §3.1 |
| Decoupled RoPE dim (d_R_h) | 64 | paper §3.1 |
| seq_len | 8192 | reused from the attention project (Ch8, *How to Scale Your Model*) — still realistic here since DeepSeek-V2 supports up to 128K context; kept for cross-project comparability (Phase 4 synthesis). Not a first-order lever for MoE comms volume the way it was for attention (dispatch is per-token, not sequence-quadratic). |
| Chip | **TPU 8i** | Google Cloud Next 2026 (GA April 22, 2026) — [technical deep dive](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive) |
| Precision | **FP4, uniform** (compute + comms, simplifying assumption) | chosen to match TPU 8i's native/headline format; optional future refinement: split dispatch-traffic precision higher (e.g. FP8) if numbers end up close |

### TPU 8i specs (used throughout)

- 288 GB HBM per chip, 8.6 TB/s HBM bandwidth
- 384 MB on-chip SRAM
- 10.1 PFLOPS FP4 compute
- Inter-chip interconnect: 19.2 Tb/s, on a new **"Boardfly" topology**
  explicitly built to cut network diameter (~56%) for MoE/reasoning
  workloads — real reference point for Phase 2's topology hypothesis later
- Collective acceleration engine (CAE) offloads all-reduce/all-gather

**Why TPU 8i over TPU v5e** (v5e was the attention project's chip, from Ch8):
the reason v5e was chosen last time — mimicking Gemmini's scale/constraints
for RTL validation — doesn't apply here, since ASTRA-sim is a topology-
agnostic network simulator, not tied to a specific small RTL generator's
physical scale. TPU 8i wins on its own merits instead: it's purpose-built for
MoE serving (Boardfly topology directly answers Phase 2's topology question),
and it gives one consistent source for both per-chip memory numbers (needed
now) and inter-chip interconnect numbers (needed for Phase 1b/1c/2), instead
of stitching together specs from two unrelated sources.

### Deployment model (resolved after a TP-sharding detour — see below)

- **Attention: data-parallel** across the 8 EP-group devices. Each device
  holds a full, unsharded copy of the attention weights and runs attention
  locally on its own slice of the batch. No cross-device communication for
  attention itself.
- **MoE FFN: expert-parallel.** After attention, each token's hidden-state
  activation (d_model=5120-dim vector) gets **dispatched** across the
  interconnect to whichever device(s) hold its top-6 routed experts (capped
  at M≤3 devices per token), computed there using that device's local expert
  weights, and the result is **combined** back to the token's home device
  (needed there for the residual stream / next layer's attention / KV cache).
  This dispatch→combine round trip is exactly what Phase 1b/1c need to
  quantify next.
- **Shared experts (2) are replicated on every device** — every token uses
  both, so replication avoids a hotspot. Consequence: shared experts
  generate **no dispatch traffic** (always computed locally); only the 160
  routed experts are subject to routing/dispatch.
- Each device's local expert count: 20 routed (160/8) + 2 replicated shared
  = **22 local experts**.

### Why data-parallel attention, not tensor-parallel (the detour)

Initial instinct was to TP-shard attention the way you'd TP-shard vanilla
MHA — sum all weight matrices, divide by 8. This breaks for MLA specifically:

- TP-by-heads works when a weight matrix's **output has a per-head axis**
  (shape contains `n_h`) — each device can own a disjoint set of heads and
  compute them fully independently, needing only a single combine step
  (all-reduce) at the very end (`W^O`).
- Three of MLA's 8 weight matrices — `W^DKV`, `W^DQ`, `W^KR` — have **no**
  `n_h` in their shape. They each produce one shared latent/vector that
  *every* head reads in full. You can't give a device 1/8 of that vector's
  dimensions — it can't compute even one complete head's output without the
  full thing, so it would need to fetch the missing pieces from other
  devices before doing anything (communication before every layer,
  defeating the point of TP).
- Resolution: sidestep the "how do you TP a shared-latent matrix" problem
  entirely by not TP-sharding attention at all — data-parallel instead (see
  above). Each device holds the full, correct 8-matrix MLA stack.

### MLA weight matrices (the corrected structure — not vanilla MHA)

Source: DeepSeek-V2 paper, §2.1.1–2.1.3, equations 1–19 (arXiv 2405.04434,
pp. 6–8).

| Matrix | Role | Shape | Per-head? |
|---|---|---|---|
| W^DKV | KV down-projection (h → compressed latent) | d_c × d | No |
| W^UK | Key up-projection | (n_h·d_h) × d_c | Yes |
| W^UV | Value up-projection | (n_h·d_h) × d_c | Yes |
| W^DQ | Query down-projection | d_c' × d | No |
| W^UQ | Query up-projection | (n_h·d_h) × d_c' | Yes |
| W^QR | Decoupled query (RoPE) | (n_h·d_R_h) × d_c' | Yes |
| W^KR | Decoupled key (RoPE), **shared across all heads** | d_R_h × d | No |
| W^O | Output projection | d × (n_h·d_h) | Yes |

Note: paper states `W^UK` can be algebraically absorbed into `W^Q` and `W^UV`
into `W^O` at inference time (avoids materializing explicit K/V) — that's a
compute/activation-memory trick at serving time, not a smaller set of
*stored* parameters. All 8 matrices above are what's trained/checkpointed and
what the weight-memory footprint below is based on.

**MLA KV cache formula** (paper's own conclusion, eq. after 2.1.3): total
cache = `(d_c + d_R_h) · l` elements — i.e., **576 elements/token/layer**,
vs. vanilla MHA/GQA's `2 · H_kv · d_head` (per-head, includes separate K and
V). No factor of 2 (only the shared compressed latent is cached; V is
reconstructed from it via up-projection at attention time), and no per-head
axis at all (that's the actual compression — not fewer heads, no head axis).

### Per-device weight footprint (arithmetic, plugged in per user request)

- Attention (8 MLA matrices, unsharded): **149,225,472 params/layer**
- FFN (22 local experts × 3×d_model×d_ff, SwiGLU-style gate/up/down):
  22 × 3×5120×1536 = **519,045,120 params/layer**
- Total: **668,270,592 params/layer**
- × 60 layers = **40,096,235,520 params ≈ 40.1B**
- At FP4 (0.5 bytes/param): **≈ 20.048 GB** of the 288GB HBM budget

**Consistency check** (unprompted, for confidence): summing FFN params
across all 162 experts (not just the 22 local ones) × 59 MoE layers, plus
attention × 60 layers, gives ≈234.5B total params vs. the paper's stated
236B — close enough to trust the formulas; the ~1.5B gap is presumably the
dense first layer's different FFN, embeddings, and output head, none of
which were modeled.

### Batch size derivation

- Remaining budget after weights: 288GB − 20.048GB = **267.95GB** for KV
  cache
- KV cache bytes/token (all 60 layers): 0.5 × 576 × 60 = **17,280
  bytes/token**
- Max total cached tokens: 267,951,882,240 / 17,280 ≈ **15,506,474**
- Max local batch @ seq_len=8192: 15,506,474 / 8192 ≈ **1,893**
- **Chosen (with safety margin for activations/router overhead/framework
  overhead): local batch = 1,024/device** (power-of-2 convention, matches
  the attention project's batch=32 style; ~46% margin below the 1,893 max)
- **This is per-device.** Total system-wide batch = 1,024 × 8 devices =
  8,192 sequences in flight, since all 8 devices' local batches
  independently feed the shared/routed expert pool via dispatch.
- Sanity check: ~59x larger local batch than the attention project's
  batch=32 (vanilla MHA/GQA, same TPU-generation-adjacent memory budget
  order of magnitude) — expected direction, since MLA's entire point is
  enabling much larger serving batches via the compressed cache. Not a red
  flag.

---

## Open items flagged for Phase 1b / 1c (not yet resolved)

- **Device-limited routing (M≤3) changes the base "uniform routing" formula
  itself**, not just the imbalance model. The spec's naive
  `(top_k × tokens)/num_experts` formula assumes a token's experts land
  wherever across all 8 devices; DeepSeek-V2 first picks top-3 *devices* by
  affinity, then top-6 *experts* within those 3. Dispatch fan-out per token
  is bounded by device count (≤3), not expert count (6). 1b's "uniform"
  derivation needs to account for this.
- **Shared experts generate no dispatch traffic** (replicated, computed
  locally) — 1b/1c should scope communication-volume derivation to the 160
  routed experts only.
- **Three balance-loss coefficients from the paper, not yet researched**:
  α1=0.003, α2=0.05, α3=0.02 — paper distinguishes expert-level,
  device-level, and *communication*-level balance losses. Which one(s) are
  the right grounding for 1c's imbalance model is an open research question
  (this is exactly the "real, published, actively-discussed" literature the
  spec's 1c asks for — not to be invented from scratch).
- **Token-dropping is training-only** — paper explicitly states no dropping
  at evaluation. Since this project's analysis is inference/serving-focused,
  this likely rules out token-dropping as an imbalance-mitigation
  assumption; imbalance would instead show up as buffering/stall pressure
  rather than dropped-token traffic reduction. Worth confirming this
  framing before building the 1c imbalance model.

---

## Phase 1b: Communication volume — uniform routing (ideal case)

### Scope decisions (resolved during derivation)

- **Regime: decode-step, not prefill.** This whole derivation is per single
  autoregressive decode step — each forward pass through a layer processes
  exactly **one new token per active sequence**, regardless of that
  sequence's context length. seq_len=8192 governs KV-cache sizing (already
  used in Phase 1a's batch derivation) but does **not** multiply into
  per-step token count here — that's exactly what Phase 1a's note ("seq_len
  is not a first-order lever for MoE comms volume") meant. A prefill-regime
  version of this derivation (tokens = sequences × seq_len) is a distinct,
  not-yet-done extension, flagged for later if wanted.
- **Dispatch/combine precision: FP4, uniform** — kept consistent with the
  Phase 1a simplifying assumption. Flagged caveat (not yet resolved,
  low-priority): real EP systems may prefer higher precision than compute
  for cross-device hops specifically, since each independent
  quantize→dequantize round trip at a network boundary introduces its own
  rounding error, vs. a local matmul's single quantize-then-accumulate
  in fp32. Worth revisiting only if a future numbers come out close to the
  ridge point.
- **FLOPs-side scope: all 8 experts/token** (2 shared + 6 routed), not just
  the 6 routed ones — decided because the compute-to-comms ratio's purpose
  is a device-level bottleneck question (does a device's compute engine or
  its network port gate execution time — the spec's `time = max(compute,
  communication)` framing), and shared-expert compute occupies the same
  physical compute engine even though it generates zero dispatch traffic.
  ("Is routing worth it" — routed-only FLOPs vs. comms — is a different,
  narrower question not used here.)

### Per-token dispatch fan-out (device-limited routing)

Verified against the primary source (arXiv 2405.04434v5, §2.2.2, not just
trusted from memory): *"for each token, we first select M devices that have
experts with the highest affinity scores in them. Then, we perform top-K
selection among experts on these M devices."* This is **structural, not
coincidental** — a token's 6 routed experts are guaranteed by construction
to live on at most 3 devices, every time (device selection happens before
expert selection, and expert selection is restricted to fall within the
selected devices). Paper also notes M≥3 is empirically equivalent to
unrestricted global top-K (no quality loss at this cap).

Home-device overlap: a token's home device (assigned by batch-sharding at
the serving layer, unrelated to the router's affinity scores, which are a
function of token content) may itself be one of the 3 selected devices, in
which case that portion of dispatch is local/free — same reason shared
experts generate no dispatch traffic. Since **1b is the uniform/ideal
control case** (worst-case reasoning belongs in 1c's imbalance model, per
the spec's own phase split), used **expected value**, not worst case:

- P(home ∈ top-3) = 3/8 → 2 remote hops
- P(home ∉ top-3) = 5/8 → 3 remote hops
- **E[remote devices/token] = (3/8)(2) + (5/8)(3) = 2.625**

### Combine mirrors dispatch

If multiple of a token's 6 routed experts land on the same device, that
device sums their weighted outputs locally and sends back **one** combined
result, not one per expert — same "pay per unique remote device" logic as
dispatch. Payload size is also identical to dispatch: the expert FFN's
down-projection (d_ff=1536 → d_model=5120) happens **locally**, before
anything crosses the network, so the combine payload is d_model-sized, same
as the dispatch payload — the down-projection is invisible to the comms
derivation.

### Bytes moved

- Payload/token = d_model × precision bytes = 5,120 × 0.5 (FP4) = **2,560
  bytes**
- Per-token-per-layer = payload × 2 (dispatch + combine) × E[remote
  devices] = 2,560 × 2 × 2.625 = **13,440 bytes**
- Token population (system-wide, one decode step): 1,024 sequences/device ×
  8 devices = **8,192 tokens** (the "×8 devices" is already baked into
  8,192 — do not multiply by 8 again)
- **Total comms/layer = 13,440 × 8,192 = 110,100,480 bytes = exactly 105
  MiB** (dispatch + combine, whole system, one decode step)

### FLOPs moved

- Per-token-per-expert (SwiGLU: gate + up, each d_model→d_ff; down,
  d_ff→d_model; standard 2×M×N×K GEMM count, M=1): 3 × 2 × d_model × d_ff =
  3 × 2 × 5,120 × 1,536 = **47,185,920 FLOPs**
- × 8 experts/token (2 shared + 6 routed) = **377,487,360 FLOPs/token**
- × 8,192 tokens = **3,092,376,453,120 FLOPs ≈ 3.09 TFLOPs/layer** (whole
  system, one decode step — same scope as the comms figure, required for a
  valid ratio)

### Ridge point and verdict

TPU 8i ICI bandwidth (19.2 Tb/s) verified independently — Google's own
primary blog post doesn't state the ICI number explicitly, but multiple
secondary sources corroborate it (e.g. "2.4 TB/s (19.2 Tb/s), roughly double
Ironwood [TPU v7]'s 1.2 TB/s"). Note the units: 19.2 **Tb/s** (bits) ÷ 8 =
**2.4 TB/s** (bytes) — comparing the raw 19.2 number directly against HBM's
8.6 TB/s without converting units makes ICI look faster than HBM, which
would be physically backwards; converted correctly, ICI (2.4 TB/s) is
well below HBM (8.6 TB/s), as expected.

- Ridge point = FLOPS ÷ ICI bandwidth = 10.1×10¹⁵ ÷ 2.4×10¹² ≈ **4,208
  FLOPs/byte**
- Workload arithmetic intensity = 3,092,376,453,120 ÷ 110,100,480 ≈
  **28,087 FLOPs/byte**
- **≈6.7× above the ridge point → decisively compute-bound**, in the
  ideal/uniform routing case.

**Caveat carried into 1c:** this compute-bound headroom is a system-wide,
ideal-case average. It does not guarantee the workload stays compute-bound
once real routing imbalance is introduced — a locally comms-bound moment
(one overloaded device) can exist even while the global average looks
compute-bound. That's the load-bearing reason the spec keeps 1b and 1c as
separate deliverables rather than one number.

---

## Next step

Phase 1b complete. Two outstanding items, per `moe_handoff.md`:

1. **Phase 0** (still not executed): build ASTRA-sim locally via Docker.
   Boilerplate/tooling, not blocking further hand-derivation, but needed
   before Phase 3 (simulation) can happen.
2. **Phase 1c** (next real 🧠 work): derive comms volume under load
   imbalance — research a realistic imbalance model from the actual MoE
   literature (capacity factor, the paper's α1/α2/α3 balance-loss
   coefficients — which one grounds the right model is still unresolved —
   and whether token-dropping, stated as training-only in the paper, is
   even a valid assumption to import into this inference-focused analysis).
   Flagged as the phase most likely to balloon in scope — pace carefully.
