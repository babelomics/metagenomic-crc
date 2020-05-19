# coding: utf-8
"""
author: Carlos Loucera
email: carlos.loucera@juntadeandalucia.es

Utiity functions.
"""
from pathlib import Path
import dotenv


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
    elif which.lower():
        path = get_raw_path()
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
    return DOTENV_FILE.joinpath("data", "processed")


def get_raw_path() -> Path:
    """[summary]

    Returns
    -------
    Path
        [description]
    """
    return DOTENV_FILE.joinpath("data", "raw")
