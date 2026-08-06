"""Intermediate KV cache pool -- Phase 0's "block until space frees" placeholder.

simpy.Container already gives FIFO-fair blocking gets for free: a request
that can't get enough bytes right now queues, and is served in arrival order
once enough space frees -- exactly the policy this project decided on, no
extra design needed.
"""

import simpy


class IntermediateKVPool:
    def __init__(self, env: simpy.Environment, capacity_bytes: float):
        self.env = env
        self.capacity_bytes = capacity_bytes
        self.container = simpy.Container(env, capacity=capacity_bytes, init=capacity_bytes)
        # (time, level) samples for a continuous-time occupancy view, alongside
        # the per-request pool_wait metric computed from Request timestamps.
        self.samples: list[tuple[float, float]] = []

    def acquire(self, n_bytes: float):
        return self.container.get(n_bytes)

    def release(self, n_bytes: float):
        return self.container.put(n_bytes)

    @property
    def level(self) -> float:
        return self.container.level

    def monitor(self, interval_s: float):
        while True:
            self.samples.append((self.env.now, self.container.level))
            yield self.env.timeout(interval_s)
