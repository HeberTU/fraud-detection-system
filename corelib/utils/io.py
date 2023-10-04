# -*- coding: utf-8 -*-
"""Input and output module.

Created on: 3/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import os
import pathlib
import pickle
from typing import Any


def dump_artifacts(obj: Any, file_path: pathlib.Path, file_name: str) -> None:
    """Dump object to pickle format.

    Args:
        obj: object
            Object to save.
        file_path: pathlib.Path
            File path
        file_name: str
            Name of the file
    """
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    with open(file=file_path / file_name, mode="wb") as handle:
        pickle.dump(obj=obj, file=handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_artifacts(file_path: pathlib.Path) -> Any:
    """Load a pickle object from memory.

    Args:
        file_path: pathlib.Path
            file path

    Returns:
        object:
            Saved object.
    """
    with open(file=file_path, mode="rb") as handle:
        obj = pickle.load(handle)

    return obj
