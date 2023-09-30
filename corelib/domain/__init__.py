# -*- coding: utf-8 -*-
"""Business Domain library.

Created on: 29/9/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from corelib.domain.data_simulator import generate_customer_profiles_list
from corelib.domain.models import CustomerProfile

__all__ = [
    "CustomerProfile",
    "generate_customer_profiles_list",
]
