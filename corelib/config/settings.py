# -*- coding: utf-8 -*-
"""Settings module.

Created on: 1/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from pathlib import Path

from pydantic import BaseSettings
from pydantic.types import DirectoryPath


class Settings(BaseSettings):
    """Project Settings."""

    PROJECT_PATH: DirectoryPath = Path(__file__).parents[1]

    # Log settings
    LOG_PATH: DirectoryPath = PROJECT_PATH / "log"
    CACH_PATH: DirectoryPath = PROJECT_PATH / ".cachedir"

    # Logging settings.
    LOG_FILE_NAME: str = "log.log"

    MAX_BYTES: int = 20 * 1024 * 1024

    MAX_DAYS: int = 365

    FILE_LOG_FORMAT: str = (
        "{time:YYYY-MM-DD HH:mm:ss} (local) ({elapsed}) | "
        "{level} | "
        "{name}:{function}:{line} - {message}"
    )


settings = Settings()
