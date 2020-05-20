# coding: utf-8
"""
author: Carlos Loucera
email: carlos.loucera@juntadeandalucia.es

Data manipulation module.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mlgut.utils import get_path


PROJECT_PATH = get_path("project")
PROCESSED_DATA_PATH = get_path("processed")
RAW_DATA_PATH = get_path("raw")


def read_metadata_frame(fpath: Path) -> pd.DataFrame:
    """[summary]

    Parameters
    ----------
    fpath : Path
        [description]

    Returns
    -------
    pd.DataFrame
        [description]
    """
    metadata = pd.read_csv(fpath, sep="\t", index_col=0)

    return metadata


def build_metadata():
    """[summary]

    Returns
    -------
    [type]
        [description]
    """
    metadata_path = RAW_DATA_PATH.joinpath("metadata")
    metadata_frames = []

    for path in metadata_path.glob("**/*subset.tsv"):
        print(path)
        frame = read_metadata_frame(path)
        analysis = "_".join(path.parent.name.split("_")[:-1])
        frame["analysis"] = analysis
        metadata_frames.append(frame)
        print(analysis, frame.shape)

    metadata = pd.concat(metadata_frames, axis=0)
    metadata["DISEASE"].fillna("X", inplace=True)
    metadata.set_index("ENA-RUN", drop=True, inplace=True)

    first_columns = metadata.columns.difference(
        pd.Index(["ENA-SAMPLE", "NGLess_Num_reads", "analysis"])
    )
    first_dict = {column: "first" for column in first_columns}
    group_var = "ENA-SAMPLE"
    special_dict = {"NGLess_Num_reads": "sum", "analysis": lambda x: "_".join(set(x))}
    col_dict = {**first_dict, **special_dict}

    metadata = metadata.groupby(group_var).agg(col_dict)

    return metadata


def read_profile_frame(path: Path) -> pd.DataFrame:
    """[summary]

    Parameters
    ----------
    path : Path
        [description]

    Returns
    -------
    pd.DataFrame
        [description]
    """
    # project-wise preprocessing
    if path.exists():
        df = pd.read_csv(path, sep="\t", index_col=0)
        if (np.array(df.shape) == 0).any():
            print("error loading project: {}".format(path))
            df = None
    else:
        print("project does not exist: {}".format(path))
        df = None

    return df


def build_features(metadata: pd.DataFrame, profiles=["KEGG_KOs", "centrifuge"]):
    """[summary]

    Parameters
    ----------
    metadata : pd.DataFrame
        [description]
    profiles : List[str], optional
        [description], by default ["KEGG_KOs", "centrifuge", "OGs"]

    Returns
    -------
    [type]
        [description]
    """
    ngless_profiles_path = RAW_DATA_PATH.joinpath("ngless_samples_profiles")
    centrifuge_profiles_path = RAW_DATA_PATH.joinpath("centrifuge_samples_profiles")
    projects = metadata.SECONDARY_STUDY_ID.unique()

    profiles_frames_dict = {profile: [] for profile in profiles}
    profiles_dict = {profile: pd.DataFrame() for profile in profiles}

    for project in projects:
        for profile in profiles:
            if profile != "centrifuge":
                fname = f"{project}_ngless_{profile}.tsv"
                fpath = ngless_profiles_path.joinpath(fname)
            else:
                fname = f"{project}_{profile}_samples.tsv"
                fpath = centrifuge_profiles_path.joinpath(fname)
            profile_frame = read_profile_frame(fpath)
            profiles_frames_dict[profile].append(profile_frame)

    for profile in profiles:
        frame_list = [
            frame for frame in profiles_frames_dict[profile] if frame is not None
        ]

        features = pd.concat(frame_list, axis=0, join="outer")
        features.fillna(0.0, inplace=True)
        profiles_dict[profile] = features

    common_index = metadata.index
    for profile in profiles:
        common_index = common_index.intersection(profiles_dict[profile].index)

    metadata = metadata.loc[common_index, :]
    for profile in profiles:
        profiles_dict[profile] = profiles_dict[profile].loc[common_index, :]

    return metadata, profiles_dict


def write_metadata(frame: pd.DataFrame):
    """[summary]

    Parameters
    ----------
    frame : pd.DataFrame
        [description]
    """
    fname = "metadata.tsv"
    fpath = PROCESSED_DATA_PATH.joinpath(fname)
    frame.to_csv(fpath, sep="\t", index_label="ENA-SAMPLE")


def write_features(profile_dict: dict):
    """[summary]

    Parameters
    ----------
    profile_dict : dict
        [description]
    """
    for profile in profile_dict.keys():
        fname = f"{profile}.tsv"
        fpath = PROCESSED_DATA_PATH.joinpath(fname)
        features = profile_dict[profile]
        features.to_csv(fpath, sep="\t", index_label="ENA-SAMPLE")


def read_metadata():    
    fname = 'metadata.tsv'
    fpath = PROCESSED_DATA_PATH.joinpath(fname)
    metadata = pd.read_csv(fpath, sep="\t", index_col=0)
    
    return metadata


def build_condition_dataset(condition, profile_name="KEGG_KOs", batch=None):
    
    metadata = read_metadata()
    
    condition_query = metadata["DISEASE"] == condition
    projects = metadata.loc[condition_query, "SECONDARY_STUDY_ID"].unique()
    project_query = metadata["SECONDARY_STUDY_ID"].isin(projects)
    
    metadata = metadata.loc[project_query, :]
    
    if batch is None:
        fname = f'{profile_name}.tsv'
    else:
        fname = f"{profile_name}_{batch}.tsv"
    
    fpath = PROCESSED_DATA_PATH.joinpath(fname)
    features = pd.read_csv(fpath, sep="\t", index_col="ENA-SAMPLE")
    
    features = features.loc[project_query, :]
    
    return features, metadata


if __name__ == "__main__":
    metadata = build_metadata()
    metadata, features_dict = build_features(metadata)
    write_metadata(metadata)
    write_features(features_dict)
