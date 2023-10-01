# -*- coding: utf-8 -*-
"""Timer module.

Created on: 1/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import functools
import time
from typing import (
    Any,
    Callable,
)

from corelib.utils.logging import get_logger

logger = get_logger()


def timer(func: Callable) -> Callable:
    """Measure the time a function takes to execute.

    Args:
        func: Callable
            Function to measure.

    Returns:
       Callable:
        Functions decorated with timing measuring artifacts.
    """

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        msg = f"Finished {func.__name__!r} in {run_time:.4f} secs"
        logger.info(msg)
        return result

    return wrapper_timer
