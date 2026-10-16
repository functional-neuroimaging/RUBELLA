"""Check that every module of the package can be imported."""
import importlib
import pkgutil

import pytest

import rubella

# every module found under the rubella package, e.g. 'rubella.basic', 'rubella.caps.main'
MODULES = [module.name for module in pkgutil.walk_packages(rubella.__path__, prefix='rubella.')]


@pytest.mark.parametrize('module_name', MODULES)
def test_import(module_name):
    importlib.import_module(module_name)
