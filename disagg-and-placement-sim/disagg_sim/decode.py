"""Decode machine + dispatcher.

Each admitted request runs as its OWN independent SimPy process generating
tokens one at a time -- not synchronized/lockstep batching across all
resident requests. Each time a request is about to generate its next token,
it reads how many requests are currently resident on ITS machine right now
(live occupancy `n`, which varies as requests join/leave) and computes that
token's service time via service_time.decode_token_time(n) fresh. This is a
state-dependent-service-rate queue, deliberately avoiding full iteration-level
lockstep scheduling (Phase 0's own stated preference).

Occupancy is a plain int counter, not a simpy.Resource: the dispatcher already
does centralized admission control (only pops from `waiting` when a machine
has room), so a generic multi-consumer Resource would be redundant -- and
worse, actively wrong. Resource.count only updates when a spawned process
reaches its `yield`, not at the synchronous moment the dispatcher decides to
admit it, so two requests dispatched to the same machine within one
_try_dispatch() call could both see stale occupancy and over-admit past the
cap. Incrementing a plain counter synchronously at admission time (before the
process is even spawned) avoids that race entirely.
"""

from disagg_sim import service_time as service_time_mod
from disagg_sim.workload import Request


class DecodeMachine:
    def __init__(self, machine_id: int, cap: int):
        self.id = machine_id
        self.cap = cap
        self.occupancy = 0
        # Event-driven (time, occupancy) samples -- one appended every time
        # occupancy actually changes (admit/complete), not on a fixed
        # interval, since it only changes at those instants. Lightweight:
        # occupancy changes far less often than the per-token timeouts do.
        self.occupancy_samples: list[tuple[float, int]] = []


class DecodeDispatcher:
    def __init__(self, env, config, machines: list[DecodeMachine], pool, on_complete=None):
        self.env = env
        self.config = config
        self.machines = machines
        self.pool = pool
        self.waiting: list[Request] = []
        self.on_complete = on_complete  # Callable[[Request], None], for metrics collection

    def submit(self, req: Request):
        self.waiting.append(req)
        self._try_dispatch()

    def _try_dispatch(self):
        while self.waiting:
            # join-shortest-queue: balances load across the machine pool,
            # consistent with the N=320 derivation's implicit even-load
            # assumption (Phase 1).
            candidate = min(self.machines, key=lambda m: m.occupancy)
            if candidate.occupancy >= candidate.cap:
                break  # every machine is at its cap
            req = self.waiting.pop(0)
            candidate.occupancy += 1  # commit synchronously, before spawning the process
            candidate.occupancy_samples.append((self.env.now, candidate.occupancy))
            req.decode_admit_time = self.env.now
            # The KV cache physically leaves the intermediate pool the moment
            # it's admitted to a decode machine -- release the bytes it was
            # holding there. Without this the pool only ever fills and never
            # drains, since PrefillDispatcher._handoff() only ever acquires.
            self.env.process(self._release_pool_bytes(req))
            self.env.process(self._run_request(req, candidate))

    def _release_pool_bytes(self, req: Request):
        yield self.pool.release(req.kv_bytes)

    def _run_request(self, req: Request, machine: DecodeMachine):
        tokens_remaining = req.output_len
        while tokens_remaining > 0:
            n = machine.occupancy  # live occupancy, read fresh each token
            t = service_time_mod.decode_token_time(n, self.config.decode_avg_context_len)
            yield self.env.timeout(t)
            tokens_remaining -= 1
        req.completion_time = self.env.now
        machine.occupancy -= 1
        machine.occupancy_samples.append((self.env.now, machine.occupancy))
        if self.on_complete is not None:
            self.on_complete(req)
        self._try_dispatch()  # a slot just freed -- let a waiting request in
