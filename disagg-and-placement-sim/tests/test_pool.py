import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import simpy

from disagg_sim.pool import IntermediateKVPool


def test_acquire_blocks_when_insufficient_and_unblocks_on_release():
    env = simpy.Environment()
    pool = IntermediateKVPool(env, capacity_bytes=100)
    events = []

    def holder():
        yield pool.acquire(80)
        events.append(("holder_acquired", env.now))
        yield env.timeout(10)
        yield pool.release(80)
        events.append(("holder_released", env.now))

    def waiter():
        yield env.timeout(1)  # ensure holder acquires first
        yield pool.acquire(50)  # only 20 free after holder -- must block until release
        events.append(("waiter_acquired", env.now))

    env.process(holder())
    env.process(waiter())
    env.run()

    kinds = [e[0] for e in events]
    assert kinds == ["holder_acquired", "holder_released", "waiter_acquired"], kinds
    waiter_time = dict((k, t) for k, t in events)["waiter_acquired"]
    assert waiter_time >= 10, "waiter should not acquire before holder releases at t=10"


def test_acquire_succeeds_immediately_when_capacity_available():
    env = simpy.Environment()
    pool = IntermediateKVPool(env, capacity_bytes=1000)
    acquired_at = []

    def proc():
        yield pool.acquire(10)
        acquired_at.append(env.now)

    env.process(proc())
    env.run()
    assert acquired_at == [0]
