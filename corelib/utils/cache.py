# -*- coding: utf-8 -*-
"""Caching module.

Created on: 1/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import functools
import hashlib
import inspect
import pickle
import types
from inspect import getsource
from typing import Callable

import numpy as np
import pandas as pd


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

        try:

            result = load_from_memory(hash_str=hash_rep, prefix=func.__name__)

        except FileNotFoundError:

            result = func(*args, **kwargs)

            dump_result(obj=result, hash_str=hash_rep, prefix=func.__name__)

        return result

    return cacher_wrapper


def dump_result(obj: object, hash_str: str, prefix: str) -> None:
    """Dump object to pickle format.

    Args:
        obj: object
            Object to save.
        hash_str: str
            Unique identifier fo the object.
        prefix: str
            Provide human-readable info on the filename generated.
    """
    file_name = ".cachedir/" + prefix + "_" + hash_str + ".pickle"
    with open(file=file_name, mode="wb") as handle:
        pickle.dump(obj=obj, file=handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_from_memory(hash_str: str, prefix: str) -> object:
    """Load a pickle object from memory.

    Args:
        hash_str: str
            Unique identifier fo the object.
        prefix: str
            Provide human-readable info on the filename generated.

    Returns:
        object:
            Saved object.
    """
    file_name = ".cachedir/" + prefix + "_" + hash_str + ".pickle"

    with open(file=file_name, mode="rb") as handle:
        obj = pickle.load(handle)

    return obj


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


def make_obj_hash(obj: object, mode: str = "fast") -> str:
    """Hash an object.

    Args:
        obj: object
            Object to be hashed.
        mode: str
            hash mode for pandas dataframes.


    Returns:
        str: hashed representation.
    """
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
                obj = obj.sample(100, random_state=42, replace=True)
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