"""SimConfig -- one instance per (lambda, pool_capacity, seed) replication."""

from dataclasses import dataclass

from disagg_sim import constants


@dataclass
class SimConfig:
    arrival_rate: float  # req/s
    pool_capacity_bytes: float
    seed: int = 0

    n_prefill_machines: int = constants.N_PREFILL_MACHINES
    n_decode_machines: int = constants.N_DECODE_MACHINES
    prefill_batch_cap: int = constants.PREFILL_BATCH_CAP
    decode_cap_per_machine: int = constants.DECODE_N_CAP
    decode_avg_context_len: float = constants.DECODE_AVG_CONTEXT_LEN

    prompt_len_mean: float = constants.PROMPT_LEN_MEAN
    prompt_len_cv: float = constants.PROMPT_LEN_CV
    output_len_mean: float = constants.OUTPUT_LEN_MEAN
    output_len_cv: float = constants.OUTPUT_LEN_CV

    warmup_completions: int = constants.WARMUP_COMPLETIONS
    target_completions: int = constants.TARGET_COMPLETIONS

    # Safety valve: hard stop if a run somehow never reaches target_completions
    # (e.g. a misconfigured cell where lambda exceeds system capacity by a lot
    # and the pool deadlocks) -- avoids a sweep replication hanging forever.
    max_sim_time_s: float = 3600.0
