# -*- coding: utf-8 -*-
"""Logging module.

Created on: 1/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
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
