# -*- coding: utf-8 -*-
"""Settings module.

Created on: 1/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from __future__ import annotations

import enum
import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic.types import DirectoryPath
from pydantic_settings import BaseSettings


class Environment(str, enum.Enum):
    """Available Environments."""

    PROD: Environment = "PROD"
    TEST: Environment = "TEST"


class LogLevels(str, enum.Enum):
    """Available DB types."""

    debug = "DEBUG"
    info = "INFO"
    important = "IMPORTANT"
    warning = "WARNING"
    error = "ERROR"


class Settings(BaseSettings):
    """Project Settings."""

    class Config:
        """Loads the dotenv file."""

        env_file: str = ".env"

    ENV: Optional[str] = Field(None, env="ENV")

    PROJECT_PATH: DirectoryPath = Path(__file__).parents[2]

    # Log settings
    LOG_PATH: DirectoryPath = PROJECT_PATH / "log"
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)

    CACH_PATH: DirectoryPath = PROJECT_PATH / ".cachedir"
    if not os.path.exists(CACH_PATH):
        os.makedirs(CACH_PATH)

    ASSETS_PATH: DirectoryPath = PROJECT_PATH / "assets" / "api"
    if not os.path.exists(ASSETS_PATH):
        os.makedirs(ASSETS_PATH)

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
