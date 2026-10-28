"""
FETCHERS FOR REFERENCE FILES DISTRIBUTED WITH RABIES (TEMPLATES, ATLASES, SEEDS)
"""
import os
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

# directory where RABIES installs its default template files
if 'XDG_DATA_HOME' in os.environ:
    RABIES_DATA_DIR = Path(os.environ['XDG_DATA_HOME']) / 'rabies'
else:
    RABIES_DATA_DIR = Path.home() / '.local/share/rabies'

# directory where RUBELLA saves the files it downloads, kept separate from the RABIES install
if 'XDG_CACHE_HOME' in os.environ:
    RUBELLA_CACHE_DIR = Path(os.environ['XDG_CACHE_HOME']) / 'rubella'
else:
    RUBELLA_CACHE_DIR = Path.home() / '.cache/rubella'

_GITHUB = 'https://github.com/CoBrALab/RABIES/releases/download/0.5.1'
_ZENODO = 'https://zenodo.org/records/19069284/files'

# remote file name -> (url, md5 checksum)
_REMOTE = {
    'DSURQE_40micron.nii.gz': (f'{_GITHUB}/DSURQE_40micron.nii.gz', '542586156f1046a6db3be119c08db9b5'),
    'DSURQE_40micron_mask.nii.gz': (f'{_GITHUB}/DSURQE_40micron_mask.nii.gz', '917591c746acf42eafc8335705c4f557'),
    'DSURQE_40micron_labels.nii.gz': (f'{_GITHUB}/DSURQE_40micron_labels.nii.gz', 'b64b8d3b08d4bace719ed4a6db6a75b5'),
    'EPICOMMON.zip': (f'{_ZENODO}/EPICOMMON.zip', '33368b18ed616431796d4dff088c2611'),
    'DSURQE_seeds.zip': (f'{_ZENODO}/DSURQE_seeds.zip', '9b31347c3715ea40397688a79979c806'),
}

# names of the pre-built RABIES seeds (same names as accepted by the RABIES --seed_list parameter)
SEED_NAMES = [
    'ACA_seed', 'AMYG_seed', 'basal_ganglia_seed', 'ECT_AI_seed', 'HIP_seed', 'HY_seed', 'MO_seed',
    'ORB_limbic_seed', 'RSP_seed', 'SS_dorsal_seed', 'SS_frontal_seed', 'THAL_seed', 'VIS_seed']

# for each space, file key -> (remote file providing it, local path relative to the data directory).
# Local paths follow the same layout as the RABIES install, and for zip archives they also
# correspond to the archive member names.
_SPACES = {
    'anat': {
        'template': ('DSURQE_40micron.nii.gz', 'DSURQE_40micron_average.nii.gz'),
        'brain_mask': ('DSURQE_40micron_mask.nii.gz', 'DSURQE_40micron_mask.nii.gz'),
        'labels': ('DSURQE_40micron_labels.nii.gz', 'DSURQE_40micron_labels.nii.gz'),
        'seeds': ('DSURQE_seeds.zip', 'DSURQE_seeds/{seed}_left.nii.gz'),
    },
    'EPI': {
        'template': ('EPICOMMON.zip', 'EPICOMMON/EPICOMMON_atlas/EPICOMMON_template.nii.gz'),
        'brain_mask': ('EPICOMMON.zip', 'EPICOMMON/EPICOMMON_atlas/EPICOMMON_brain_mask.nii.gz'),
        'labels': ('EPICOMMON.zip', 'EPICOMMON/EPICOMMON_atlas/EPICOMMON_labels.nii.gz'),
        'seeds': ('DSURQE_seeds.zip', 'DSURQE_seeds/EPICOMMON_resampled/{seed}_left_EPICOMMON_resampled.nii.gz'),
    },
}


def fetch_rabies_templates(space='anat', cache_dir=None, verbose=True):
    """Fetch the RABIES mouse template files and pre-built seeds for a given space.

    Two template spaces are available:

    * ``'anat'``: the DSURQE 40 micron anatomical template, which is the default RABIES
      commonspace template.
    * ``'EPI'``: the EPICOMMON template, which is the commonspace template used by RABIES
      with ``--bold_only``.

    Each file is first searched for inside the RABIES install directory
    (``$XDG_DATA_HOME/rabies``, or ``~/.local/share/rabies``), and used from there if found.
    Otherwise, it is searched for inside the RUBELLA cache directory (``$XDG_CACHE_HOME/rubella``,
    or ``~/.cache/rubella``), and downloaded there if missing. Downloaded files are never saved
    inside the RABIES install directory, so that they cannot conflict with a RABIES install.
    Downloads are verified with a checksum.

    Parameters
    ----------
    space : str
        The template space, among ``'anat'`` or ``'EPI'``.

    cache_dir : str or pathlib.Path, optional
        Directory where missing files are downloaded, in place of the default RUBELLA cache
        directory.

    verbose : bool
        Whether to print which files are being downloaded.

    Returns
    -------
    dict
        Paths to the files in the selected space: ``'template'`` (the template image),
        ``'brain_mask'``, ``'labels'`` (the anatomical parcellation), and ``'seeds'``, a
        dictionary of seed name -> seed mask path, for each seed in ``SEED_NAMES``.

    Examples
    --------
    >>> from rubella.datasets import fetch_rabies_templates
    >>> epi = fetch_rabies_templates('EPI')
    >>> epi['seeds']['ACA_seed']
    """
    if space not in _SPACES:
        raise ValueError(f"space must be among {list(_SPACES.keys())}, but '{space}' was provided.")
    cache_dir = Path(cache_dir) if cache_dir is not None else RUBELLA_CACHE_DIR

    files = {}
    for key, (remote, local) in _SPACES[space].items():
        if key == 'seeds':
            files['seeds'] = {seed: _get_file(remote, local.format(seed=seed), cache_dir, verbose)
                              for seed in SEED_NAMES}
        else:
            files[key] = _get_file(remote, local, cache_dir, verbose)
    return files


def _get_file(remote, local, cache_dir, verbose):
    # returns the path to a file, from the RABIES install if available, otherwise from the cache
    for data_dir in [RABIES_DATA_DIR, cache_dir]:
        if (data_dir / local).is_file():
            return str(data_dir / local)
    _download(remote, cache_dir, verbose)
    return str(cache_dir / local)


def _download(remote, cache_dir, verbose):
    url, md5 = _REMOTE[remote]
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_dir / f'{remote}.part' # downloading to a temporary file avoids keeping incomplete files
    if verbose:
        print(f'Downloading {url} into {cache_dir}')
    with urllib.request.urlopen(url) as response, open(tmp, 'wb') as f:
        shutil.copyfileobj(response, f)
    if _md5(tmp) != md5:
        tmp.unlink()
        raise IOError(f'Checksum mismatch for {url}; the download is corrupted or the remote file has changed.')

    # all local paths provided by this remote file, across spaces
    local_paths = [local.format(seed=seed) for space_files in _SPACES.values()
                   for (r, local) in space_files.values() if r == remote
                   for seed in (SEED_NAMES if '{seed}' in local else [None])]
    if remote.endswith('.zip'):
        # only the needed members are extracted (e.g. the EPICOMMON archive also contains large transforms)
        with zipfile.ZipFile(tmp) as z:
            for member in local_paths:
                z.extract(member, cache_dir)
        tmp.unlink()
    else:
        tmp.rename(cache_dir / local_paths[0])


def _md5(file):
    h = hashlib.md5()
    with open(file, 'rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()
