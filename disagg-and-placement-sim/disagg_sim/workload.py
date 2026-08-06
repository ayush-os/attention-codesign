"""Request shape + arrival process.

Poisson arrivals; prompt/output length lognormal, means DistServe-anchored
(512/64, notes.md Phase 0). No variance was ever specified upstream for this
distribution (confirmed via full-file search of notes.md) -- CV=1.0 for both,
independently sampled, is a named config default (constants.PROMPT_LEN_CV /
OUTPUT_LEN_CV), flagged explicitly as an approximation carried forward from a
genuine unresolved upstream gap, not a derived number.
"""

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from disagg_sim.config import SimConfig
from disagg_sim.constants import KV_BYTES_PER_TOKEN


@dataclass
class Request:
    id: int
    arrival_time: float
    prompt_len: int
    output_len: int
    kv_bytes: float  # prompt_len * KV_BYTES_PER_TOKEN -- varies per request (SS5.5 of the plan)

    # Timestamps filled in as the request progresses through the system.
    # Each *_wait property below derives the queueing time from these.
    prefill_dispatch_time: float = None
    prefill_done_time: float = None
    pool_acquired_time: float = None
    decode_admit_time: float = None
    completion_time: float = None

    @property
    def prefill_wait(self) -> float:
        return self.prefill_dispatch_time - self.arrival_time

    @property
    def pool_wait(self) -> float:
        return self.pool_acquired_time - self.prefill_done_time

    @property
    def decode_admission_wait(self) -> float:
        return self.decode_admit_time - self.pool_acquired_time

    @property
    def end_to_end_latency(self) -> float:
        return self.completion_time - self.arrival_time


def lognormal_params(mean: float, cv: float) -> tuple[float, float]:
    sigma = math.sqrt(math.log(1 + cv**2))
    mu = math.log(mean) - sigma**2 / 2
    return mu, sigma


def sample_request_shape(rng: np.random.Generator, config: SimConfig) -> tuple[int, int]:
    mu_p, sig_p = lognormal_params(config.prompt_len_mean, config.prompt_len_cv)
    mu_o, sig_o = lognormal_params(config.output_len_mean, config.output_len_cv)
    prompt_len = max(1, round(rng.lognormal(mu_p, sig_p)))
    output_len = max(1, round(rng.lognormal(mu_o, sig_o)))
    return prompt_len, output_len


def arrival_process(env, config: SimConfig, rng: np.random.Generator, on_arrival: Callable[[Request], None]):
    req_id = 0
    while True:
        interarrival = rng.exponential(1.0 / config.arrival_rate)
        yield env.timeout(interarrival)
        prompt_len, output_len = sample_request_shape(rng, config)
        req = Request(
            id=req_id,
            arrival_time=env.now,
            prompt_len=prompt_len,
            output_len=output_len,
            kv_bytes=prompt_len * KV_BYTES_PER_TOKEN,
        )
        req_id += 1
        on_arrival(req)
