# Sphinx configuration for the RUBELLA API documentation.
# Full option list: https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# make the rubella package importable for autodoc without installing it
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
project = 'RUBELLA'
author = 'Gozzi lab'
# read the version from rubella/__version__.py, as setup.py does, without importing the package
about = {}
with open(os.path.join(os.path.abspath('..'), 'rubella', '__version__.py')) as f:
    exec(f.read(), about)
release = about['__version__']

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',     # pulls API docs from docstrings
    'sphinx.ext.autosummary', # summary tables of functions/classes in api.md
    'sphinx.ext.napoleon',    # parses numpydoc-style sections (Parameters, Returns, ...)
    'sphinx.ext.viewcode',    # adds [source] links to each documented object
    'myst_parser',            # only needed if you write pages in Markdown; remove otherwise
]

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- autodoc / napoleon ------------------------------------------------------
# All third-party dependencies are mocked so the docs build in a minimal
# environment (only sphinx, myst-parser and the theme need to be installed).
autodoc_mock_imports = ['rabies', 'nilearn', 'SimpleITK', 'sklearn', 'scipy', 'seaborn',
                        'numpy', 'pandas', 'matplotlib']
autodoc_member_order = 'bysource'   # keep the order of the source file
napoleon_google_docstring = False   # the docstrings are numpydoc only
napoleon_numpy_docstring = True

# -- HTML output -------------------------------------------------------------
html_theme = 'sphinx_rtd_theme'     # same look as RABIES; requires `pip install sphinx-rtd-theme`
html_title = f'RUBELLA {release}'