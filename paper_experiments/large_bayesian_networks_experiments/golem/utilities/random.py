import random
from typing import Optional

import numpy as np

from golem.core.log import default_log


class RandomStateHandler:
    MODEL_FITTING_SEED = 0

    @staticmethod
    def log_random_state():
        py_state = random.getstate()
        np_state = np.random.get_state()
        log = default_log(RandomStateHandler.__class__.__name__)
        log.debug(f'Random State: random.getstate() follows...\n{py_state}')
        log.debug(f'Random State: numpy.random.get_state() follows...\n{np_state}')

    def __init__(self, seed: Optional[int] = None):
        if seed is None:
            seed = RandomStateHandler.MODEL_FITTING_SEED
        self._seed = seed
        self._old_seed = None

    def __enter__(self):
        self._old_np_state = np.random.get_state()
        self._old_state = random.getstate()

        np.random.seed(self._seed)
        random.seed(self._seed)
        return self._seed

    def __exit__(self, exc_type, exc_value, exc_traceback):
        np.random.set_state(self._old_np_state)
        random.setstate(self._old_state)
