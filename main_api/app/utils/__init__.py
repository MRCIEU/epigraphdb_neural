import time
from typing import Any, Callable, Dict, List

import numpy as np
from loguru import logger


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


def vector_empty(vector: List[float]) -> bool:
    arr = np.array(vector)
    empty = np.count_nonzero(arr) == 0
    return empty
