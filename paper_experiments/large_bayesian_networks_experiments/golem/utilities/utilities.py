from typing import Optional

import numpy as np
from joblib import cpu_count

import random
from golem.utilities.random import RandomStateHandler


def determine_n_jobs(n_jobs=-1, logger=None):
    cpu_num = cpu_count()
    if n_jobs > cpu_num:
        n_jobs = cpu_num
    elif n_jobs <= 0:
        if n_jobs <= -cpu_num - 1 or n_jobs == 0:
            raise ValueError(f"Unproper `n_jobs` = {n_jobs}. "
                             f"`n_jobs` should be between ({-cpu_num}, {cpu_num}) except 0")
        n_jobs = cpu_num + 1 + n_jobs
    if logger:
        logger.info(f"Number of used CPU's: {n_jobs}")
    return n_jobs


def urandom_mock(n):
    # os.random is the source of random used in the uuid library
    # normally, it's „true“ random, but to stabilize tests,
    # seeded `random` library is used instead.
    return bytes(random.getrandbits(8) for _ in range(n))


def set_random_seed(seed: Optional[int]):
    """ Sets random seed for evaluation of models. """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        RandomStateHandler.MODEL_FITTING_SEED = seed
