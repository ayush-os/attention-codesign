"""Follow-up exploration after the main sweep showed zero pool contention at
every tested size (9.73-233 GB): (1) does that hold well past the estimated
system ceiling (~4,138 req/s), not just barely above it (4,500 was the only
above-ceiling lambda tested); (2) where between 84 MB (confirmed to cause
near-total collapse in the earlier diagnostic) and 9.73 GB (zero contention)
does pool contention actually start to show up.

Deliberately short max_sim_time_s (60s, vs. the default 3600s) and a lighter
completion target -- this is exploratory, meant to fail fast on pathological
cells rather than burn wall-clock time confirming what's already known
(sub-100MB collapses badly). Not part of the validated sweep pipeline.
"""

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from disagg_sim import metrics as metrics_mod
from disagg_sim.config import SimConfig
from disagg_sim.run_single import run_replication

LAMBDAS = [4500, 6000, 8000, 10000]
POOL_BYTES = {
    "200MB": 200e6,
    "500MB": 500e6,
    "1GB": 1e9,
    "2GB": 2e9,
    "4GB": 4e9,
    "9.73GB": 9.73e9,  # smallest size from the original sweep, as a reference point
}
N_SEEDS = 3
TARGET_COMPLETIONS = 3000
WARMUP_COMPLETIONS = 300
MAX_SIM_TIME_S = 60.0


def build_configs():
    configs = []
    for lam in LAMBDAS:
        for label, nbytes in POOL_BYTES.items():
            for seed in range(N_SEEDS):
                configs.append(
                    (
                        label,
                        SimConfig(
                            arrival_rate=lam,
                            pool_capacity_bytes=nbytes,
                            seed=seed,
                            target_completions=TARGET_COMPLETIONS,
                            warmup_completions=WARMUP_COMPLETIONS,
                            max_sim_time_s=MAX_SIM_TIME_S,
                        ),
                    )
                )
    return configs


def run_one(args) -> pd.DataFrame:
    label, config = args
    reqs = run_replication(config)
    if len(reqs) == 0:
        # Hit the max_sim_time_s safety valve before even clearing warm-up --
        # a real signal (system effectively deadlocked), not just missing data.
        return pd.DataFrame(
            [
                {
                    "lambda": config.arrival_rate,
                    "pool_label": label,
                    "pool_bytes": config.pool_capacity_bytes,
                    "seed": config.seed,
                    "incomplete": True,
                }
            ]
        )
    records = [metrics_mod.request_to_record(r) for r in reqs]
    df = pd.DataFrame(records)
    df["lambda"] = config.arrival_rate
    df["pool_label"] = label
    df["pool_bytes"] = config.pool_capacity_bytes
    df["seed"] = config.seed
    df["incomplete"] = False
    return df


def main():
    configs = build_configs()
    print(f"Running {len(configs)} replications (max {MAX_SIM_TIME_S}s each)...")
    start = time.time()
    with ProcessPoolExecutor() as ex:
        dfs = list(ex.map(run_one, configs))
    print(f"Done in {time.time() - start:.1f}s")

    raw = pd.concat(dfs, ignore_index=True)
    raw.to_csv("results/threshold_raw.csv", index=False)

    incomplete = raw[raw["incomplete"]].groupby(["lambda", "pool_label"]).size()
    if len(incomplete):
        print("\nCells that hit the 60s safety valve before finishing (effectively collapsed):")
        print(incomplete.to_string())

    complete = raw[~raw["incomplete"]]

    def p99(s):
        return s.quantile(0.99)

    summary = (
        complete.groupby(["lambda", "pool_label", "pool_bytes"])
        .agg(
            n_requests=("id", "count"),
            frac_pool_blocked=("pool_wait", lambda s: (s > 1e-9).mean()),
            pool_wait_mean=("pool_wait", "mean"),
            pool_wait_p99=("pool_wait", p99),
            pool_wait_max=("pool_wait", "max"),
            decode_admit_wait_mean=("decode_admission_wait", "mean"),
            e2e_latency_mean=("end_to_end_latency", "mean"),
        )
        .reset_index()
        .sort_values(["lambda", "pool_bytes"])
    )
    summary.to_csv("results/threshold_summary.csv", index=False)
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
