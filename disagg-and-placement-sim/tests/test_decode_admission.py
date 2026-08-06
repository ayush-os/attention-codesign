import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import simpy

from disagg_sim.config import SimConfig
from disagg_sim.decode import DecodeDispatcher, DecodeMachine
from disagg_sim.pool import IntermediateKVPool
from disagg_sim.workload import Request


def test_admission_never_exceeds_cap():
    env = simpy.Environment()
    pool = IntermediateKVPool(env, capacity_bytes=1e15)
    machine = DecodeMachine(machine_id=0, cap=10)
    config = SimConfig(arrival_rate=1.0, pool_capacity_bytes=1e15, n_decode_machines=1, decode_cap_per_machine=10)
    dispatcher = DecodeDispatcher(env, config, [machine], pool)

    for i in range(50):
        req = Request(id=i, arrival_time=0.0, prompt_len=512, output_len=5, kv_bytes=1.0)
        dispatcher.submit(req)
        assert machine.occupancy <= 10, f"occupancy {machine.occupancy} exceeded cap after submitting request {i}"

    env.run()


def test_pool_bytes_released_on_admission():
    env = simpy.Environment()
    pool = IntermediateKVPool(env, capacity_bytes=100)
    machine = DecodeMachine(machine_id=0, cap=10)
    config = SimConfig(arrival_rate=1.0, pool_capacity_bytes=100, n_decode_machines=1, decode_cap_per_machine=10)
    dispatcher = DecodeDispatcher(env, config, [machine], pool)

    # Manually check out bytes the way PrefillDispatcher._handoff does, then
    # confirm admission releases them back.
    def acquire_then_submit():
        yield pool.acquire(60)
        assert pool.level == 40
        req = Request(id=0, arrival_time=0.0, prompt_len=512, output_len=1, kv_bytes=60)
        dispatcher.submit(req)

    env.process(acquire_then_submit())
    env.run()
    assert pool.level == 100, f"expected pool fully drained back to 100, got {pool.level}"


def test_join_shortest_queue_balances_across_machines():
    env = simpy.Environment()
    pool = IntermediateKVPool(env, capacity_bytes=1e15)
    machines = [DecodeMachine(machine_id=i, cap=100) for i in range(3)]
    config = SimConfig(arrival_rate=1.0, pool_capacity_bytes=1e15, n_decode_machines=3, decode_cap_per_machine=100)
    dispatcher = DecodeDispatcher(env, config, machines, pool)

    for i in range(30):
        req = Request(id=i, arrival_time=0.0, prompt_len=512, output_len=1000, kv_bytes=1.0)
        dispatcher.submit(req)

    occupancies = [m.occupancy for m in machines]
    assert occupancies == [10, 10, 10], f"expected even load-balancing, got {occupancies}"
