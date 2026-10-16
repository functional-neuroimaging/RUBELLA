"""Check that the installed dependencies match the requirements declared in setup.py."""
import ast
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest
from packaging.requirements import Requirement

SETUP_FILE = Path(__file__).parent.parent / 'setup.py'


def read_setup_requirements():
    """Return the REQUIRED list from setup.py, read as text (setup.py is not executed)."""
    tree = ast.parse(SETUP_FILE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == 'REQUIRED' for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("No REQUIRED list was found in setup.py.")


@pytest.mark.parametrize('requirement', read_setup_requirements())
def test_dependency_version(requirement):
    req = Requirement(requirement)
    try:
        installed_version = version(req.name)
    except PackageNotFoundError:
        pytest.fail(f"{req.name} is required by setup.py but is not installed.")
    assert req.specifier.contains(installed_version, prereleases=True), \
        f"{req.name} {installed_version} is installed, but setup.py requires {requirement}."
