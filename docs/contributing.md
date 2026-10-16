# Contributing new code to RUBELLA

## Basic instructions for using Git

### Getting started with basic git commands
```sh
# create your own local version of the repository
$ git clone https://github.com/functional-neuroimaging/RUBELLA.git
$ cd RUBELLA # move into your copy of the git repository
$ git status # tells you what branch you are on, and currently tracked/untracked files
  On branch main
  Untracked files:
    (use "git add <file>..." to include in what will be committed)
          contributing.md
          dependencies.txt
$ git pull origin main # update your local main branch to match the online repository, i.e. if new code was committed online
```

### Steps to follow when committing new code
```sh
# VERY IMPORTANT - create a new branch for your own dev, this will prevent introducing potential bugs on the main branch, where everybody else is working
$ git branch my_branch_for_XYZ
# change your git branch - you will now only modify this new branch, and not the main branch
$ git checkout my_branch_for_XYZ
###NOW APPLY YOUR CHANGES###
$ git status # check what you have changed so far
  On branch my_branch_for_XYZ
  Untracked files:
    (use "git add <file>..." to include in what will be committed)
          my_new_function.py

$ git add my_new_function.py # add new files/folders to the package, on your current branch

$ git status # now it is tracked
  On branch my_branch_for_XYZ
  Changes to be committed:
    (use "git restore --staged <file>..." to unstage)
          new file:   my_new_function.py
# run the test suite from the repository root to check that nothing is broken (requires `pip install pytest` once)
$ pytest tests/
# commit your changes
$ git commit -m "the new script my_new_function.py does XYZ"
$ git status # all tracked changes are commited
  On branch python_package
  nothing to commit, working tree clean
# push your commit(s) to the online repository
$ git push origin my_branch_for_XYZ
# you are done for local steps, the rest happens on GitHub, where we can merge your branch with the main branch
```

## How to document a function properly with a docstring

The [API reference](api) is generated automatically from the docstrings in the code, which must
follow the [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) format. To keep the
documentation consistent across the package, each function is ideally accompanied with a 
properly-formated docstring following this example:

```python
def my_new_function(timeseries_dict, img_id_list, brain_mask_file, threshold=0.5, verbose=True):
    """Compute XYZ for each image ID within a timeseries_dict.

    A longer description of what the function does, how it does it, and any assumption about
    the inputs. Refer to related functions with :func:`rubella.basic.seed_based_FC`.

    Parameters
    ----------
    timeseries_dict : dict
        Dictionary with ``img_id_list`` as keys, and each entry is the associated 2D timeseries
        array (time by voxel), e.g. :attr:`rubella.data_obj.DataObject.timeseries_dict`.

    img_id_list : list of str
        List of image IDs that are keys for ``timeseries_dict``.

    brain_mask_file : str or pathlib.Path
        Full path to the brain mask Nifti file.

    threshold : float, default=0.5
        Description of the parameter, and of the effect of its possible values.

    verbose : bool, default=True
        Whether to print the image ID for each iteration.

    Returns
    -------
    XYZ_dict : dict
        Dictionary with ``img_id_list`` as keys, and each entry is the associated XYZ array of
        shape (n_voxels,).

    Examples
    --------
    >>> XYZ_dict = my_new_function(data_obj.timeseries_dict, data_obj.img_id_list,
    ...                            data_obj.brain_mask_file, threshold=0.5)
    """
```

AI is very useful for generating a well-formatted first draft of a docstring - you can point it to this
documentation and your new code, and it should provide a decent first draft for your review.

These are the detailed instructions:
- **Summary line**: start the docstring with a one-line summary on the same line as the opening
  `"""`, followed by a blank line. This line is displayed in the summary tables of the API reference.
- **Sections**: use the `Parameters` and `Returns` sections, and optionally `Examples` and `Notes`,
  each underlined with dashes of the same length as the section title.
- **Types**: write standard Python types after the parameter name, e.g. `str or pathlib.Path`,
  `list of str`, `dict`, `numpy.ndarray`. Optional parameters end with their default value,
  e.g. `bool, default=False`.
- **Code formatting**: use double backticks for parameter names and literal values,
  e.g. ` ``img_id_list`` ` or ` ``'correlation'`` `. Single backticks render as plain italics.
- **Links**: refer to other RUBELLA objects with `` :func:`rubella.basic.seed_based_FC` ``,
  `` :class:`DataObject` ``, `` :meth:`load_censoring_mask` `` or `` :attr:`img_id_list` ``,
  which become clickable links in the documentation.
- **Methods that modify an object**: a method that sets attributes on its object returns `self`.
  Describe the new attributes once in the `Attributes` section of the class docstring, and only
  name them in the method description, e.g. "The masks are stored in the :attr:\`censoring_mask\`
  attribute."

**Importantly**, before committing you should do the following:

- add the new function to the list of its module in `docs/api.md`, so that it appears in the
  summary table of the API reference;
- add a test in the `tests/` folder (the fake data in `tests/conftest.py` can be reused), and
  run `pytest tests/`;
- check that the documentation builds without warnings, using
  `sphinx-build -b html docs docs/_build/html` (requires `pip install -r docs/requirements.txt`).