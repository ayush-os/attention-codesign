"""Service-time formulas -- the file every other correctness claim depends on.

General roofline `time = max(FLOPs/peak_compute, bytes/HBM_bandwidth)` applies
to all three terms (attention, QKVO, FFN), both phases -- not just QKVO/FFN.
QKVO and FFN are weight-stationary: fixed weight-load bytes regardless of
batch size/occupancy (batch-invariant), so they have a flat memory-bound floor
at low T and a linear compute-bound regime above the crossover (~T=296,
identical crossover point for both terms since they share the same
FLOPs/bytes ratio -- 21.4%). Attention's bytes/FLOPs both scale with tokens
processed -- no crossover, but NOT assumed memory-bound a priori; the max()
lets it land wherever the real numbers put it (prefill attention actually
lands marginally compute-bound at the real seq_len=512 operating point, per
notes.md SS2.3 -- this correctly reproduces that finding rather than
overriding it with an assumption).
"""

from disagg_sim.constants import (
    D_HEAD,
    FFN_FLOPS_PER_TOKEN_LAYER,
    FFN_WEIGHT_BYTES_LAYER,
    HBM_BANDWIDTH_BPS,
    N_HEADS,
    N_KV_HEADS,
    N_LAYERS,
    PEAK_COMPUTE_FLOPS,
    PRECISION_BYTES,
    QKVO_FLOPS_PER_TOKEN_LAYER,
    QKVO_WEIGHT_BYTES_LAYER,
)


def _roofline_time(flops: float, bytes_moved: float) -> float:
    return max(flops / PEAK_COMPUTE_FLOPS, bytes_moved / HBM_BANDWIDTH_BPS)


def _weight_stationary_time(tokens: float, flops_per_token: float, weight_bytes: float) -> float:
    compute_t = tokens * flops_per_token / PEAK_COMPUTE_FLOPS
    mem_t = weight_bytes / HBM_BANDWIDTH_BPS
    return max(compute_t, mem_t)


def prefill_attention_flops_bytes(prompt_lens: list[int]) -> tuple[float, float]:
    """Self-attention (seq_len_q = seq_len_kv = L_i), summed per-sequence.

    NOT a function of aggregate T = sum(prompt_lens) alone: FLOPs scale as
    L_i^2 per sequence, which is convex in L, so using mean(L) for the whole
    batch would understate total attention time via Jensen's inequality once
    prompt lengths vary per request (they do, under the lognormal workload
    model in workload.py). Summing per-sequence is exact and reduces to the
    fixed-512 reference case exactly when all L_i = 512.
    """
    flops = sum(4 * N_HEADS * D_HEAD * (l**2) for l in prompt_lens)
    bytes_ = sum(l * D_HEAD * PRECISION_BYTES * (2 * N_HEADS + 2 * N_KV_HEADS) for l in prompt_lens)
    return flops, bytes_


def prefill_batch_layer_time(prompt_lens: list[int]) -> float:
    tokens = sum(prompt_lens)
    attn_flops, attn_bytes = prefill_attention_flops_bytes(prompt_lens)
    attn_t = _roofline_time(attn_flops, attn_bytes)
    qkvo_t = _weight_stationary_time(tokens, QKVO_FLOPS_PER_TOKEN_LAYER, QKVO_WEIGHT_BYTES_LAYER)
    ffn_t = _weight_stationary_time(tokens, FFN_FLOPS_PER_TOKEN_LAYER, FFN_WEIGHT_BYTES_LAYER)
    return attn_t + qkvo_t + ffn_t


def prefill_batch_time(prompt_lens: list[int]) -> float:
    """Full per-batch prefill service time (seconds), all 80 layers."""
    return N_LAYERS * prefill_batch_layer_time(prompt_lens)


def decode_attention_flops_bytes(n: int, avg_context_len: float) -> tuple[float, float]:
    """seq_len_q = 1 (one new token per resident request), seq_len_kv = avg_context_len."""
    flops = n * 4 * N_HEADS * D_HEAD * 1 * avg_context_len
    bytes_ = n * D_HEAD * PRECISION_BYTES * (2 * N_HEADS * 1 + 2 * N_KV_HEADS * avg_context_len)
    return flops, bytes_


def decode_token_layer_time(n: int, avg_context_len: float) -> float:
    attn_flops, attn_bytes = decode_attention_flops_bytes(n, avg_context_len)
    attn_t = _roofline_time(attn_flops, attn_bytes)
    qkvo_t = _weight_stationary_time(n, QKVO_FLOPS_PER_TOKEN_LAYER, QKVO_WEIGHT_BYTES_LAYER)
    ffn_t = _weight_stationary_time(n, FFN_FLOPS_PER_TOKEN_LAYER, FFN_WEIGHT_BYTES_LAYER)
    return attn_t + qkvo_t + ffn_t


def decode_token_time(n: int, avg_context_len: float = 544.0) -> float:
    """Time (seconds) for one token-step for a request, given live occupancy n."""
    return N_LAYERS * decode_token_layer_time(n, avg_context_len)
