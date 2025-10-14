# Licensed under a 3-clause BSD style license - see LICENSE.rst

from .Quaternion import *

__version__ = '3.5.2'


def test(*args, **kwargs):
    '''
    Run py.test unit tests.
    '''
    import os
    import pytest
    return pytest.main([os.path.dirname(__file__)]).value
