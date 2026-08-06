"""Direct answer to: are decode machines actually underutilized, or just
under-admitted (lots of spare admission slots, but the requests actually
running are keeping the compute busy)? Those are different questions --
admission-cap headroom (already established) says nothing on its own about
whether occupancy sits above or below the ~296 compute-bound crossover.

Time-weighted average occupancy from occupancy_samples (event-driven, exact
-- not a periodic poll) directly answers this per machine, per lambda.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import simpy

from disagg_sim import constants
from disagg_sim.config import SimConfig
from disagg_sim.system import build_system

CROSSOVER_N = 296  # notes.md SS2.5


def time_weighted_stats(samples: list[tuple[float, int]], end_time: float):
    if not samples:
        return 0.0, 0, 0.0
    times = [t for t, _ in samples] + [end_time]
    occs = [o for _, o in samples]
    weighted_sum = 0.0
    frac_above_crossover_time = 0.0
    for i in range(len(occs)):
        dt = times[i + 1] - times[i]
        weighted_sum += occs[i] * dt
        if occs[i] >= CROSSOVER_N:
            frac_above_crossover_time += dt
    total_time = end_time - times[0]
    mean_occ = weighted_sum / total_time if total_time > 0 else 0.0
    max_occ = max(occs)
    frac_above = frac_above_crossover_time / total_time if total_time > 0 else 0.0
    return mean_occ, max_occ, frac_above


def run_one(lam: float, seed: int = 0, sim_time: float = 5.0):
    env = simpy.Environment()
    rng = np.random.default_rng(seed)
    config = SimConfig(arrival_rate=lam, pool_capacity_bytes=constants.ROUND_UNIT_BYTES, seed=seed)
    system = build_system(env, config, rng)
    env.run(until=sim_time)
    print(f"\nlambda={lam}, sim_time={sim_time}s, cap={config.decode_cap_per_machine}")
    for m in system.decode_machines:
        mean_occ, max_occ, frac_above = time_weighted_stats(m.occupancy_samples, sim_time)
        print(
            f"  machine {m.id}: mean_occ={mean_occ:6.1f}  max_occ={max_occ:4d}  "
            f"frac_time_above_crossover({CROSSOVER_N})={frac_above:.1%}  cap_utilization={mean_occ / m.cap:.1%}"
        )


if __name__ == "__main__":
    for lam in [500, 2000, 3500, 4500, 6000, 8000, 10000]:
        run_one(lam, seed=0, sim_time=5.0)
