import time
from pathlib import Path
from typing import Any, Callable, Dict

from loguru import logger


def find_project_root(anchor_file: str = "environment.yml") -> Path:
    cwd = Path.cwd()
    test_dir = cwd
    prev_dir = None
    while prev_dir != test_dir:
        if (test_dir / anchor_file).exists():
            return test_dir
        prev_dir = test_dir
        test_dir = test_dir.parent
    return cwd


def chunk_process(
    chunk_idx: int, size: int, func: Callable, kwargs: Dict
) -> Any:
    logger.info(f"#{chunk_idx}/{size}")
    return func(**kwargs)


def timeit(method: Callable):
    def timed(*args, **kwargs):
        start_time = time.time()
        result = method(*args, **kwargs)
        finish_time = time.time()
        elapsed_mins = round((finish_time - start_time) / 60, 2)
        logger.info(
            "{method} finished in {elapsed_mins} mins.".format(
                method=method.__name__, elapsed_mins=elapsed_mins
            )
        )
        return result

    return timed
