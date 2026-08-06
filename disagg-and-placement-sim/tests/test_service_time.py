"""Regression tests pinning the two reference throughput numbers from
../disagg_and_placement_notes.md SS2 exactly. If these don't match, the
formula translation is wrong -- not the workload. Every other number in
this simulator depends on these being right.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from disagg_sim import service_time


def test_prefill_reference_throughput():
    # batch=32 sequences, all at the anchor prompt length (512).
    t = service_time.prefill_batch_time([512] * 32)
    throughput = 32 / t
    assert abs(throughput - 142.70) < 0.01, f"got {throughput!r} req/s/chip, expected ~142.70"


def test_decode_reference_throughput():
    # live occupancy n=320, the derived HBM-capacity ceiling.
    t = service_time.decode_token_time(320, avg_context_len=544.0)
    token_throughput = 320 / t
    assert abs(token_throughput - 53_157.6) < 5, f"got {token_throughput!r} tok/s, expected ~53,157.6"
    request_throughput = token_throughput / 64
    assert abs(request_throughput - 830.59) < 0.1, f"got {request_throughput!r} req/s/chip, expected ~830.59"


def test_prefill_batch_time_matches_reference_layer_time():
    # notes.md SS2.9: combined/layer at batch=32 ~= 2,803.2 us; x80 layers ~= 224.26 ms.
    per_layer_us = service_time.prefill_batch_layer_time([512] * 32) * 1e6
    assert abs(per_layer_us - 2803.2) < 1.0, f"got {per_layer_us!r} us/layer, expected ~2803.2"


def test_decode_token_time_matches_reference_layer_time():
    # notes.md SS2.9: combined/layer at N=320 ~= 75.25 us.
    per_layer_us = service_time.decode_token_layer_time(320, 544.0) * 1e6
    assert abs(per_layer_us - 75.25) < 0.1, f"got {per_layer_us!r} us/layer, expected ~75.25"


def test_prefill_attention_is_marginally_compute_bound():
    # notes.md SS2.3: real finding is a ~1.55x compute-bound margin, not memory-bound.
    from disagg_sim.constants import HBM_BANDWIDTH_BPS, PEAK_COMPUTE_FLOPS

    flops, bytes_ = service_time.prefill_attention_flops_bytes([512] * 32)
    compute_t = flops / PEAK_COMPUTE_FLOPS
    mem_t = bytes_ / HBM_BANDWIDTH_BPS
    margin = compute_t / mem_t
    assert abs(margin - 1.55) < 0.02, f"got {margin!r}x margin, expected ~1.55x compute-bound"


def test_decode_ffn_crossover_near_296():
    # notes.md SS2.5: FFN/QKVO regime crossover is at N ~ 296. Isolate the
    # weight-stationary (FFN) term specifically -- attention grows ~linearly
    # with N throughout (no flat region), so it must be excluded to test the
    # actual claim: FFN alone went 41.00us (N=32) -> 44.65us (N=320), ~9%
    # growth despite a 10x batch increase -- the "free batching" finding.
    from disagg_sim.constants import FFN_FLOPS_PER_TOKEN_LAYER, FFN_WEIGHT_BYTES_LAYER

    ffn_32 = service_time._weight_stationary_time(32, FFN_FLOPS_PER_TOKEN_LAYER, FFN_WEIGHT_BYTES_LAYER) * 1e6
    ffn_320 = service_time._weight_stationary_time(320, FFN_FLOPS_PER_TOKEN_LAYER, FFN_WEIGHT_BYTES_LAYER) * 1e6
    assert abs(ffn_32 - 41.00) < 0.1, f"got {ffn_32!r} us at N=32, expected ~41.00 (memory floor)"
    assert abs(ffn_320 - 44.65) < 0.1, f"got {ffn_320!r} us at N=320, expected ~44.65 (just past crossover)"


def test_decode_attention_grows_roughly_linearly():
    # By contrast, attention has no batch-invariant floor -- growth from N=32
    # to N=320 (10x) should be close to 10x, not flat.
    from disagg_sim.constants import HBM_BANDWIDTH_BPS, PEAK_COMPUTE_FLOPS

    flops_32, bytes_32 = service_time.decode_attention_flops_bytes(32, 544.0)
    flops_320, bytes_320 = service_time.decode_attention_flops_bytes(320, 544.0)
    t_32 = max(flops_32 / PEAK_COMPUTE_FLOPS, bytes_32 / HBM_BANDWIDTH_BPS) * 1e6
    t_320 = max(flops_320 / PEAK_COMPUTE_FLOPS, bytes_320 / HBM_BANDWIDTH_BPS) * 1e6
    assert abs(t_32 - 2.103) < 0.01, f"got {t_32!r} us at N=32, expected ~2.103"
    ratio = t_320 / t_32
    assert abs(ratio - 10.0) < 0.1, f"got {ratio!r}x growth N=32->320, expected ~10x (linear)"
