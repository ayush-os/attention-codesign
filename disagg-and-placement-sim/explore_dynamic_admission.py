"""Compares the hard-cap admission policy Phase 3 actually chose (N=320,
sized off worst-case per-request HBM reservation) against the looser
dynamic-admission policy Phase 3 only ever throughput-sensitivity-checked
(N=4,558, sized off average-case reservation). Phase 3 found throughput is
essentially identical either way (N=320 already sits past the compute-bound
crossover) -- what it couldn't check is queueing latency, since that only
exists once real arrivals/admission are simulated. That's what this measures.

Pool capacity is held fixed at the original "1 round" default (~38.92 GB,
already confirmed safe from the threshold exploration) so pool contention
can't confound the admission-cap comparison.
"""

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from disagg_sim import constants
from disagg_sim import metrics as metrics_mod
from disagg_sim.config import SimConfig
from disagg_sim.run_single import run_replication

LAMBDAS = [500, 2000, 3500, 4500, 6000, 8000, 10000]
DECODE_CAPS = {"hard_cap_320": 320, "dynamic_4558": 4558}
N_SEEDS = 5
TARGET_COMPLETIONS = 5000
WARMUP_COMPLETIONS = 500
MAX_SIM_TIME_S = 90.0


def build_configs():
    configs = []
    for lam in LAMBDAS:
        for label, cap in DECODE_CAPS.items():
            for seed in range(N_SEEDS):
                configs.append(
                    (
                        label,
                        SimConfig(
                            arrival_rate=lam,
                            pool_capacity_bytes=constants.ROUND_UNIT_BYTES,  # fixed, already-safe size
                            decode_cap_per_machine=cap,
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
        return pd.DataFrame(
            [{"lambda": config.arrival_rate, "policy": label, "seed": config.seed, "incomplete": True}]
        )
    records = [metrics_mod.request_to_record(r) for r in reqs]
    df = pd.DataFrame(records)
    df["lambda"] = config.arrival_rate
    df["policy"] = label
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
    raw.to_csv("results/dynamic_admission_raw.csv", index=False)

    incomplete = raw[raw["incomplete"]].groupby(["lambda", "policy"]).size()
    if len(incomplete):
        print("\nCells that hit the safety valve before finishing:")
        print(incomplete.to_string())

    complete = raw[~raw["incomplete"]]

    def p95(s):
        return s.quantile(0.95)

    def p99(s):
        return s.quantile(0.99)

    summary = (
        complete.groupby(["lambda", "policy"])
        .agg(
            n_requests=("id", "count"),
            pool_wait_mean=("pool_wait", "mean"),
            decode_admit_wait_mean=("decode_admission_wait", "mean"),
            decode_admit_wait_p95=("decode_admission_wait", p95),
            decode_admit_wait_p99=("decode_admission_wait", p99),
            e2e_latency_mean=("end_to_end_latency", "mean"),
            e2e_latency_p95=("end_to_end_latency", p95),
            e2e_latency_p99=("end_to_end_latency", p99),
        )
        .reset_index()
        .sort_values(["lambda", "policy"])
    )
    summary.to_csv("results/dynamic_admission_summary.csv", index=False)
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
