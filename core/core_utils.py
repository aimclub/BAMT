from contextlib import contextmanager
from timeit import default_timer
from pathlib import Path


@contextmanager
def elapsed_timer(lbl, is_print):
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end - start
    if is_print:
        print(f'{lbl}: {round(end - start, 2)} sec')


def project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent
