import os
from pathlib import Path
from typing import Optional, Iterable

from golem.core.log import default_log
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory


class HistoryReader:
    """Simplifies reading a bunch of histories from single directory."""

    def __init__(self, save_path: Optional[Path] = None):
        self.log = default_log(self)
        self.save_path = save_path or Path("results")
        self.save_path.mkdir(parents=True, exist_ok=True)

    def load_histories(self) -> Iterable[OptHistory]:
        """Iteratively loads saved histories one-by-ony."""
        num_histories = 0
        total_individuals = 0
        for history_path in HistoryReader.traverse_histories(self.save_path):
            history = OptHistory.load(history_path)
            num_histories += 1
            total_individuals += sum(map(len, history.generations))
            yield history

        if num_histories == 0 or total_individuals == 0:
            raise ValueError(f'Could not load any individuals.'
                             f'Possibly, path {self.save_path} does not exist or is empty.')
        else:
            self.log.info(f'Loaded {num_histories} histories '
                          f'with {total_individuals} individuals in total.')

    @staticmethod
    def traverse_histories(path) -> Iterable[Path]:
        if path.exists():
            # recursive traversal of the save directory
            for root, dirs, files in os.walk(path):
                for history_filename in files:
                    if history_filename.startswith('history'):
                        full_path = Path(root) / history_filename
                        yield full_path
