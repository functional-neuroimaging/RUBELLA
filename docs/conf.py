# Sphinx configuration for the RUBELLA API documentation.
# Full option list: https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
project = 'RUBELLA'
author = 'Gozzi lab'
release = '0.0.1'  # fallback if the package is not installed in the docs env

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',     # pulls API docs from docstrings
    'sphinx.ext.napoleon',    # parses numpydoc-style sections (Parameters, Returns, ...)
    'sphinx.ext.viewcode',    # adds [source] links to each documented object
    'myst_parser',            # only needed if you write pages in Markdown; remove otherwise
]

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- autodoc / napoleon ------------------------------------------------------
# Heavy or hard-to-install dependencies are mocked so the docs build without them.
# numpy, pandas and matplotlib should be installed normally.
autodoc_mock_imports = ['rabies', 'nilearn', 'SimpleITK']
autodoc_member_order = 'bysource'   # keep the order of the source file
napoleon_google_docstring = False   # the docstrings are numpydoc only
napoleon_numpy_docstring = True

# -- HTML output -------------------------------------------------------------
html_theme = 'sphinx_rtd_theme'     # same look as RABIES; requires `pip install sphinx-rtd-theme`
html_title = f'RUBELLA {release}'