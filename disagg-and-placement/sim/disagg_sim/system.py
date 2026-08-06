"""Wires the pool, prefill dispatcher, decode dispatcher/machines, and arrival
process together into one runnable system.
"""

from dataclasses import dataclass

import numpy as np

from disagg_sim.config import SimConfig
from disagg_sim.decode import DecodeDispatcher, DecodeMachine
from disagg_sim.pool import IntermediateKVPool
from disagg_sim.prefill import PrefillDispatcher
from disagg_sim.workload import Request, arrival_process


@dataclass
class System:
    env: object
    pool: IntermediateKVPool
    prefill_dispatcher: PrefillDispatcher
    decode_dispatcher: DecodeDispatcher
    decode_machines: list[DecodeMachine]


def build_system(env, config: SimConfig, rng: np.random.Generator, on_complete=None) -> System:
    pool = IntermediateKVPool(env, config.pool_capacity_bytes)
    decode_machines = [DecodeMachine(i, config.decode_cap_per_machine) for i in range(config.n_decode_machines)]
    decode_dispatcher = DecodeDispatcher(env, config, decode_machines, pool, on_complete=on_complete)
    prefill_dispatcher = PrefillDispatcher(env, config, pool, decode_dispatcher)

    env.process(arrival_process(env, config, rng, on_arrival=prefill_dispatcher.submit))

    return System(env, pool, prefill_dispatcher, decode_dispatcher, decode_machines)
