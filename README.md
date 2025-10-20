# RUBELLA: Repository of Utilitaries for Brain Experiments Low Level Analyses.

## Files

* `rubella/`          ...the software package
* `contributing.md`   ...instructions for committing new code
* `environment.yml`   ...a python environment file with all required dependencies
* `setup.py`          ...install the python package so it can be imported

## Python setup

```sh
# install miniconda/anaconda which handles python environment
# then you can install mamba, which is faster than conda for creating an environment, conda install can be very slow
conda install -n base -c conda-forge mamba
mamba env create -f environment.yml # the .yml file will create a 'rubella' environment will all required dependencies
conda activate rubella # activate your python environment
python setup.py install # this will install the rubella package, which can then be imported in python
# you're set, now you can run python with the package installed
```
