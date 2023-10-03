# -*- coding: utf-8 -*-
"""Caching module.

Created on: 1/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import functools
import hashlib
import inspect
import types
from inspect import getsource
from typing import Callable

import joblib
import numpy as np
import pandas as pd

from corelib import config
from corelib.utils.io import (
    dump_artifacts,
    load_artifacts,
)


def cacher(func: Callable) -> Callable:
    """Cache function / method result.

    Args:
        func: Callable
            Function that we want to cache

    Returns:
        Callable:
            Decorated function.
    """

    @functools.wraps(func)
    def cacher_wrapper(*args, **kwargs) -> object:
        """Cache strategy.

        Args:
            *args: func args
            **kwargs: func kwargs

        Returns:
            object:
                Function result.
        """
        args_hash = make_obj_hash(args)

        kwargs_hash = make_obj_hash(kwargs)

        func_hash = make_obj_hash(func.__name__)

        hash_rep = hash_function(args_hash + kwargs_hash + func_hash)

        file_path = config.settings.CACH_PATH
        file_name = func.__name__ + "_" + hash_rep + ".pickle"

        try:

            result = load_artifacts(file_path=file_path / file_name)

        except FileNotFoundError:

            result = func(*args, **kwargs)

            dump_artifacts(
                obj=result, file_path=file_path, file_name=file_name
            )

        return result

    return cacher_wrapper


def hash_function(obj: object) -> str:
    """Get the hash representation of a python object.

    Args:
        obj: object
            Object to hash.

    Returns:
        str:
            hashed representation on the provided object.
    """
    hasher = hashlib.sha256()
    hasher.update(repr(obj).encode())
    hash_value = hasher.hexdigest()

    return hash_value


def make_obj_hash(
    obj: object, mode: str = "fast", is_training: bool = False
) -> str:
    """Hash an object.

    Args:
        obj: object
            Object to be hashed.
        mode: str
            hash mode for pandas dataframes.
        is_training: bool
            If True  joblib will be used.


    Returns:
        str: hashed representation.
    """
    if is_training:
        return joblib.hash(obj)

    hash_value = 42

    try:

        if isinstance(obj, (set, tuple, list)):
            hash_value = hash_function(
                tuple(
                    [make_obj_hash(e) for e in obj]
                    + [make_obj_hash(obj.__class__.__name__)]
                )
            )
        elif isinstance(obj, dict):
            new_o = {}
            for k, v in obj.items():
                try:
                    new_o[k] = make_obj_hash(v)
                except TypeError:
                    pass
            hash_value = hash_function(
                tuple(
                    sorted(frozenset(new_o.items()))
                    + [make_obj_hash(obj.__class__.__name__)]
                )
            )
        elif isinstance(obj, np.ndarray):
            hash_value = hash_function(str(obj))
        elif isinstance(obj, pd.DataFrame):
            if mode == "fast" and not obj.empty:
                obj = obj.sample(n=100, random_state=42, replace=True)
            index = tuple(obj.index)
            columns = tuple(obj.columns)
            values = tuple(make_obj_hash(x) for x in obj.values)
            hash_value = hash_function(tuple([index, columns, values]))
        elif isinstance(obj, pd.Series):
            index = tuple(obj.index)
            values = tuple(make_obj_hash(x) for x in obj.values)
            hash_value = hash_function(tuple([index, values]))
        elif isinstance(obj, types.FunctionType) or inspect.ismethod(obj):
            hash_value = hash_function(getsource(obj))
        elif isinstance(obj, types.BuiltinFunctionType):
            hash_value = hash_function(obj.__name__)
        elif (
            inspect.isclass(obj)
            or inspect.ismodule(obj)
            or inspect.getmodule(obj)
        ):
            class_dict = {}
            hashables = (int, str, float, bool, list, dict, tuple, set)

            def is_primitive(thing):
                return isinstance(thing, hashables)

            def has_sourcecode(thing):
                try:
                    getsource(thing)
                    return True
                except Exception:
                    return False

            for attr in dir(obj):
                gattr = getattr(obj, attr)
                if (
                    (inspect.ismethod(gattr) and has_sourcecode(gattr))
                    or inspect.isclass(gattr)
                    or is_primitive(gattr)
                    or (inspect.isfunction(gattr) and has_sourcecode(gattr))
                ) and not attr.startswith("__"):
                    class_dict[attr] = getattr(obj, attr)
            class_dict["class"] = obj.__class__.__name__
            class_dict["__name__"] = getattr(obj, "__name__", "")
            hash_value = make_obj_hash(class_dict)
        elif not obj:
            hash_value = hash_function(str(obj))
        else:
            hash_value = hash_function(obj)

        return hash_value

    except RecursionError:
        return "RecursionError"
    except TypeError:
        try:
            return hash_function(tuple(hash_value))
        except Exception as e:
            raise e
