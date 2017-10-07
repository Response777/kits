# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import time
import functools

def log(flag = True):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            results = func(*args, **kwargs)
            end_time   = time.time()
            if(flag):
                print('%s: %.2f' % (func.__name__, end_time - start_time))
            return results
        return wrapper
    return decorator
