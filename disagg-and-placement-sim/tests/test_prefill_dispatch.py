import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import simpy

from disagg_sim.config import SimConfig
from disagg_sim.pool import IntermediateKVPool
from disagg_sim.prefill import PrefillDispatcher
from disagg_sim.workload import Request


class _RecordingDecodeDispatcher:
    def __init__(self):
        self.received: list[Request] = []

    def submit(self, req):
        self.received.append(req)


def _make(n_machines=1, cap=32):
    env = simpy.Environment()
    pool = IntermediateKVPool(env, capacity_bytes=1e15)
    config = SimConfig(
        arrival_rate=1.0, pool_capacity_bytes=1e15, n_prefill_machines=n_machines, prefill_batch_cap=cap
    )
    decode = _RecordingDecodeDispatcher()
    dispatcher = PrefillDispatcher(env, config, pool, decode)
    return env, dispatcher, decode


def test_batch_capped_at_prefill_batch_cap():
    env, dispatcher, decode = _make(n_machines=1, cap=32)
    for i in range(100):
        req = Request(id=i, arrival_time=0.0, prompt_len=512, output_len=64, kv_bytes=1.0)
        dispatcher.submit(req)
    env.run()
    # 100 requests, cap=32 -> batches of 32, 32, 32, 4 -- all must have been handed off eventually
    assert len(decode.received) == 100
    # every batch's dispatch time should be shared within the batch, and no
    # batch should exceed the cap: reconstruct batches via dispatch_time grouping
    from collections import Counter

    batch_sizes = Counter(r.prefill_dispatch_time for r in decode.received)
    assert all(size <= 32 for size in batch_sizes.values()), batch_sizes


def test_small_batch_does_not_wait_for_full_cap():
    """A machine with only a few requests queued should still dispatch --
    prefill's compute-bound regime means no throughput cost to a small batch,
    and Phase 0's design never asked for a batch-formation wait."""
    env, dispatcher, decode = _make(n_machines=1, cap=32)
    for i in range(3):
        req = Request(id=i, arrival_time=0.0, prompt_len=512, output_len=64, kv_bytes=1.0)
        dispatcher.submit(req)
    env.run()
    assert len(decode.received) == 3
    # service time for a batch of 3 should be strictly less than a batch of 32
    from disagg_sim.service_time import prefill_batch_time

    assert prefill_batch_time([512] * 3) < prefill_batch_time([512] * 32)


def test_multiple_idle_machines_do_not_duplicate_or_drop_requests():
    env, dispatcher, decode = _make(n_machines=5, cap=32)
    n = 137
    for i in range(n):
        req = Request(id=i, arrival_time=0.0, prompt_len=512, output_len=64, kv_bytes=1.0)
        dispatcher.submit(req)
    env.run()
    ids = sorted(r.id for r in decode.received)
    assert ids == list(range(n)), "every request should be handed off exactly once, none duplicated or dropped"
