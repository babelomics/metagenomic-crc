# coding: utf-8
"""
author: Carlos Loucera
email: carlos.loucera@juntadeandalucia.es

Utiity functions.
"""
from pathlib import Path
import dotenv
from ete3 import NCBITaxa



DOTENV_FILE = dotenv.find_dotenv()


def get_path(which: str) -> Path:
    """[summary]

    Parameters
    ----------
    which : str
        [description]

    Returns
    -------
    Path
        [description]

    Raises
    ------
    NotImplementedError
        [description]
    """
    if which.lower() == "project":
        path = get_project_path()
    elif which.lower() == "processed":
        path = get_processed_path()
    elif which.lower() == "raw":
        path = get_raw_path()
    elif which.lower() == "results":
        path = get_results_path()
    else:
        raise NotImplementedError

    return path


def get_project_path() -> Path:
    """[summary]

    Returns
    -------
    Path
        [description]
    """
    return Path(DOTENV_FILE).parent


def get_processed_path() -> Path:
    """[summary]

    Returns
    -------
    Path
        [description]
    """
    project_path = get_project_path()
    path = project_path.joinpath("data", "processed")
    path.mkdir(parents=True, exist_ok=True)

    return path


def get_raw_path() -> Path:
    """[summary]

    Returns
    -------
    Path
        [description]
    """
    project_path = get_project_path()
    path = project_path.joinpath("data", "raw")
    path.mkdir(parents=True, exist_ok=True)

    return path


def get_results_path() -> Path:
    """[summary]

    Returns
    -------
    Path
        [description]
    """
    project_path = get_project_path()
    path = project_path.joinpath("data", "results")
    path.mkdir(parents=True, exist_ok=True)

    return path

def get_ncbi() -> NCBITaxa:
    # builds database on first call
    ncbi = NCBITaxa()

    return ncbi