# -*- coding: utf-8 -*-
"""Settings module.

Created on: 1/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import enum
import os
from pathlib import Path

from pydantic.types import DirectoryPath
from pydantic_settings import BaseSettings


class LogLevels(str, enum.Enum):
    """Available DB types."""

    debug = "DEBUG"
    info = "INFO"
    important = "IMPORTANT"
    warning = "WARNING"
    error = "ERROR"


class Settings(BaseSettings):
    """Project Settings."""

    PROJECT_PATH: DirectoryPath = Path(__file__).parents[2]

    # Log settings
    LOG_PATH: DirectoryPath = PROJECT_PATH / "log"
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)

    CACH_PATH: DirectoryPath = PROJECT_PATH / ".cachedir"
    if not os.path.exists(CACH_PATH):
        os.makedirs(CACH_PATH)

    # Logging settings.
    LOG_FILE_NAME: str = "log.log"

    MAX_BYTES: int = 20 * 1024 * 1024

    MAX_DAYS: int = 365

    FILE_LOG_FORMAT: str = (
        "{time:YYYY-MM-DD HH:mm:ss} (local) ({elapsed}) | "
        "{level} | "
        "{name}:{function}:{line} - {message}"
    )

    LOG_LEVEL: LogLevels = LogLevels.info


settings = Settings()
