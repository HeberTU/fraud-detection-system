# -*- coding: utf-8 -*-
"""API entrypoint.

Created on: 4/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from fastapi import FastAPI

from corelib.entrypoints.routes import get_router


def get_api() -> FastAPI:
    """Get API."""
    app = FastAPI
    app.include_router(get_router())
    return app
