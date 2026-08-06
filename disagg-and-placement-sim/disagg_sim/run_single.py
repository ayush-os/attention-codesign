"""Run one (lambda, pool_capacity, seed) replication to completion.

Sized by completed-request count, not fixed wall-clock simulated duration --
at high lambda, per-token SimPy processes generate a large aggregate event
volume, and a fixed-duration run risks an unpredictable event count. A
max_sim_time_s safety valve guards against a misconfigured cell (e.g. lambda
exceeding achievable system throughput) hanging forever instead of completing.
"""

import warnings

import numpy as np
import simpy

from disagg_sim.config import SimConfig
from disagg_sim.system import System, build_system
from disagg_sim.workload import Request


def _run(config: SimConfig) -> tuple[list[Request], System]:
    env = simpy.Environment()
    rng = np.random.default_rng(config.seed)
    completed: list[Request] = []
    done_event = env.event()
    target_total = config.warmup_completions + config.target_completions

    def on_complete(req: Request):
        completed.append(req)
        if len(completed) >= target_total and not done_event.triggered:
            done_event.succeed()

    system = build_system(env, config, rng, on_complete=on_complete)

    def safety_timeout():
        yield env.timeout(config.max_sim_time_s)
        if not done_event.triggered:
            done_event.succeed()

    env.process(safety_timeout())
    env.run(until=done_event)

    if len(completed) < target_total:
        warnings.warn(
            f"replication seed={config.seed} lambda={config.arrival_rate} "
            f"pool={config.pool_capacity_bytes:.3g}B only reached {len(completed)}/{target_total} "
            f"completions before max_sim_time_s={config.max_sim_time_s}s -- system likely overloaded "
            "for this config (lambda may exceed achievable throughput)."
        )

    return completed, system


def run_replication(config: SimConfig) -> list[Request]:
    """Backward-compatible entry point: post-warmup completed requests only."""
    completed, _system = _run(config)
    return completed[config.warmup_completions :]


def run_replication_with_system(config: SimConfig) -> tuple[list[Request], System]:
    """Like run_replication, but also returns the built System so callers can
    inspect machine-level diagnostics (e.g. DecodeMachine.occupancy_samples)
    that don't exist per-request. Requests are still warm-up-trimmed; the
    occupancy samples on system.decode_machines are NOT trimmed (they're
    exact event-driven history for the whole run) -- callers that want a
    fair steady-state occupancy read should filter samples to
    t >= the warm-up cutoff time themselves (the last-discarded request's
    completion_time is a reasonable proxy for that cutoff).
    """
    completed, system = _run(config)
    return completed[config.warmup_completions :], system
