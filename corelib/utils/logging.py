# -*- coding: utf-8 -*-
"""Logging module.

Created on: 1/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from typing import (
    Any,
    Dict,
)

import loguru

from corelib import config


def get_logger() -> loguru._Logger:
    """Configure logger.

    Returns:
        loguru._Logger:
            project logger.
    """
    logger = loguru.logger

    logger.add(
        sink=config.settings.LOG_PATH / config.settings.LOG_FILE_NAME,
        format=config.settings.FILE_LOG_FORMAT,
        rotation=config.settings.MAX_BYTES,
        retention=config.settings.MAX_DAYS,
        level=config.settings.LOG_LEVEL,
    )
    return logger


def format_section(title: str, data: Dict[str, Any]) -> str:
    """Formats a section title and its corresponding data for readability.

    Args:
        title str:
            The title or name of the section.
        data Dict[str, Any]:
            A dictionary containing key-value pairs to be formatted.

    Returns:
        str: A formatted string representation of the section.
    """
    max_key_length = max(len(key) for key in data.keys())
    return f"\n{title}:\n" + "\n".join(
        [
            f"    {key}: {' ' * (max_key_length - len(key))} {value}"
            for key, value in data.items()
        ]
    )


def log_model_results(logger: loguru._Logger, results: Dict[str, Any]) -> None:
    """Logs the model's results.

    Results include scores and estimator parameters in a human-readable format
    using a logger.

    Args:
        logger loguru._Logger:
            The logger instance used to log the formatted results.
        results Dict[str, Any]:
         A dictionary containing results' sections, including 'scores',
         'estimator_params', and 'hashed_data'.

    Returns:
        None
    """
    log_str = format_section(
        "Scores", {k: round(v, 3) for k, v in results["scores"].items()}
    )
    log_str += format_section(
        "Estimator Parameters", results["estimator_params"]
    )
    log_str += f"\nHashed Data: {results['hashed_data']}"

    logger.info(log_str)
