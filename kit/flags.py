# -*- coding: utf-8 -*-
# File: flags.py
# Author: Tianjian Jiang <jiangtj13@gmail.com>

import argparse

_DEBUG = False
def debug(mode = True):
    _DEBUG = mode

_global_parser = argparse.ArgumentParser()

class _Flags(object):
    """ A tf.flags style global argparser based on argparser """
    def __init__(self):
        # TODO: replace __dict__ & find a better solution to cope with infinite recursion
        self.__dict__['_flags'] = {}
        self.__dict__['_parsed'] = False

    def _parse(self, args = None):
        args, remained_args = _global_parser.parse_known_args(args = args)
        self.__dict__['_flags'].update(vars(args))
        self.__dict__['_parsed'] = True
        if(_DEBUG): print(remained_args)

    def __getattr__(self, name):
        if(not self.__dict__['_parsed']):
            self._parse()
        return self.__dict__['_flags'][name]

    def __setattr__(self, name, value):
        self.__dict__['_flags'][name] = value


FLAGS = _Flags()

def _define(flagtype):
    def wrapper(flag_name, 
                default_value,
                doc_string):
        _global_parser.add_argument('--' + flag_name,
                                    default = default_value,
                                    help = doc_string,
                                    type = flagtype)
    return wrapper

DEFINE_string  = _define(str)
DEFINE_integer = _define(int)
DEFINE_float   = _define(float)
DEFINE_bool    = _define(bool)
