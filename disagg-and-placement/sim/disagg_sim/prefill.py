"""Prefill dispatcher: grabs up to `prefill_batch_cap` queued requests the
instant a machine frees, recomputes service time for the ACTUAL batch size
taken (not fixed at the cap) -- prefill's compute-bound regime means a
smaller batch just takes proportionally less time, no reason to wait for a
full batch to form.

Single-threaded dispatcher (not a raw simpy.Store with multiple consumers) so
that when several machines go idle at once, dispatch order is deterministic
and one machine can't race another for the same pending requests -- SimPy's
cooperative single-threaded execution makes this trivially safe as long as
state mutation + the next yield happen without an intervening await, which
_try_dispatch() respects (it never yields).
"""

from disagg_sim import service_time as service_time_mod
from disagg_sim.workload import Request


class PrefillDispatcher:
    def __init__(self, env, config, pool, decode_dispatcher):
        self.env = env
        self.config = config
        self.pool = pool
        self.decode_dispatcher = decode_dispatcher
        self.pending: list[Request] = []
        self.idle_machines: list[int] = list(range(config.n_prefill_machines))

    def submit(self, req: Request):
        self.pending.append(req)
        self._try_dispatch()

    def _machine_freed(self, machine_id: int):
        self.idle_machines.append(machine_id)
        self._try_dispatch()

    def _try_dispatch(self):
        while self.idle_machines and self.pending:
            machine_id = self.idle_machines.pop(0)
            cap = self.config.prefill_batch_cap
            batch = self.pending[:cap]
            self.pending = self.pending[cap:]
            self.env.process(self._run_batch(machine_id, batch))

    def _run_batch(self, machine_id: int, batch: list[Request]):
        dispatch_time = self.env.now
        for r in batch:
            r.prefill_dispatch_time = dispatch_time
        service_time = service_time_mod.prefill_batch_time([r.prompt_len for r in batch])
        yield self.env.timeout(service_time)
        done_time = self.env.now
        for r in batch:
            r.prefill_done_time = done_time
            self.env.process(self._handoff(r))
        self._machine_freed(machine_id)

    def _handoff(self, req: Request):
        yield self.pool.acquire(req.kv_bytes)  # blocks (FIFO) if pool is full
        req.pool_acquired_time = self.env.now
        self.decode_dispatcher.submit(req)
