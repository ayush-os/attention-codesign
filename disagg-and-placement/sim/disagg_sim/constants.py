"""Hardware/model constants for the dense (Llama-3-70B, TPU 8i, FP4) leg.

All values confirmed exact against disagg-and-placement/notes.md
(SS1.3, SS2.2, SS2.9) during planning -- see that file for derivations.
"""

# Model (dense Llama-3-70B)
N_LAYERS = 80
D_MODEL = 8192
D_FF = 28672
N_HEADS = 64
N_KV_HEADS = 8
D_HEAD = 128
PRECISION_BYTES = 0.5  # FP4

# TPU 8i, FP4
PEAK_COMPUTE_FLOPS = 10.1e15  # FLOPs/s
HBM_BANDWIDTH_BPS = 8.6e12  # bytes/s

# Weight-stationary term constants (per layer), notes.md SS2.4/SS2.9
FFN_FLOPS_PER_TOKEN_LAYER = 3 * 2 * D_MODEL * D_FF  # 1,409,286,144
FFN_WEIGHT_BYTES_LAYER = 3 * D_MODEL * D_FF * PRECISION_BYTES  # 352,321,536
QKVO_FLOPS_PER_TOKEN_LAYER = 301_989_888
QKVO_WEIGHT_BYTES_LAYER = 75_497_472

# KV cache
KV_BYTES_PER_TOKEN = 81_920  # summed across all 80 layers, GQA, FP4

# System scale
N_PREFILL_MACHINES = 29
N_DECODE_MACHINES = 5

# Inherited from the attention project, never independently re-derived as a
# real TPU-8i capacity limit for THIS project (notes.md SS2.8 flags this
# explicitly). Treat as a stated assumption, not load-bearing hardware truth.
PREFILL_BATCH_CAP = 32

# This one IS a real derived HBM-capacity number (Phase 1, notes.md SS1.5).
DECODE_N_CAP = 320

# Decode's fixed context-length point estimate (Phase 0/2's own approximation:
# avg_prompt_len + avg_output_len/2 = 512 + 64/2 = 544). See sim/README.md /
# the implementation plan for the richer per-request alternative, deliberately
# not built in this pass.
DECODE_AVG_CONTEXT_LEN = 544.0

# Workload model (DistServe-anchored means; CV is a genuinely unresolved
# upstream gap -- see workload.py for the flagged default).
PROMPT_LEN_MEAN = 512.0
OUTPUT_LEN_MEAN = 64.0
PROMPT_LEN_CV = 1.0
OUTPUT_LEN_CV = 1.0

# Pool capacity sweep, anchored to a "round" = worst-case simultaneous
# full-batch completion across all prefill machines.
KV_HANDOFF_BYTES_ANCHOR = int(PROMPT_LEN_MEAN * KV_BYTES_PER_TOKEN)  # 41,943,040 B = 40 MiB
ROUND_UNIT_BYTES = N_PREFILL_MACHINES * PREFILL_BATCH_CAP * KV_HANDOFF_BYTES_ANCHOR  # ~36.25 GiB
POOL_CAPACITY_MULTIPLIERS = [0.25, 1, 2, 4, 6]

# Arrival rate sweep (req/s). System ceiling ~= min(142.70*29, 830.59*5) ~= 4,138.
ARRIVAL_RATES = [500, 2000, 3500, 4500]

# Sweep/replication defaults (Law & Kelton replication/deletion method,
# pragmatic fixed-count variant -- see plan for reasoning).
N_SEEDS = 5
WARMUP_COMPLETIONS = 2_000
TARGET_COMPLETIONS = 20_000
