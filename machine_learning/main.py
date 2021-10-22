#!/usr/bin/env python
# coding: utf-8
"""
author: Carlos Loucera
email: carlos.loucera@juntadeandalucia.es

Main processing and training module.
"""
import pathlib
import random
import subprocess
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from mlgut import analysis, datasets, models, train

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def build_data_sources(profiles=("KEGG_KOs", "centrifuge", "OGs"), ext="jbl"):
    """[summary]
    """
    metadata = datasets.build_metadata()
    metadata, features_dict = datasets.build_features(metadata, profiles)
    datasets.write_metadata(metadata, ext=ext)
    datasets.write_features(features_dict, ext=ext)


def main(
    condition,
    profile_name,
    build_data=True,
    sync=True,
    debug=True,
    ext="jbl",
    path=None,
):
    """Main routine to perform the training.

    Parameters
    ----------
    condition : str like
        Disease code.
    profile_name : str like
        Metagenomic profile name.
    build_data : bool, optional
        If True build and dump the processed files, by default True
    sync : bool, optional
        Synchronize data processed with the bioinformatics pipeline, by default True
    """
    if sync:
        print("Sync data.")
        subprocess.run(["sh", "mlgut/sync_data.sh"], check=True)
    if build_data:
        print("Build Data sources.")
        build_data_sources(ext=ext)
    if not debug:
        filter_warnings()
    if profile_name is not None:
        train_interpreter(condition, profile_name, ext=ext, save_path=path)


def filter_warnings():
    """Filter warnings when not debugging.
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)


def train_interpreter(condition, profile_name, ext, save_path):
    """Training helper function.

    Parameters
    ----------
    condition : str like
        Disease code.
    profile_name : str like
        Metagenomics profile name.
    """

    print(f"Building datasets for {condition} condition and profile {profile_name}")
    features, metadata = datasets.build_condition_dataset(
        condition, profile_name, ext=ext
    )
    features = datasets.filter_egg(features)

    # TODO: filter it with a CLI option
    # tax_id = "9606"
    # print(tax_id in features.columns)
    # features = features.drop(tax_id, axis=1)
    # features2, _ = datasets.build_condition_dataset(condition, "centrifuge")
    # features = pd.concat((features, features2), axis=1)
    model = models.get_model(profile_name)
    print(model)

    print("\t Cross-project analysis.")
    train.perform_crossproject_analysis(
        features, metadata, model, profile_name, condition, save=save_path
    )

    print("\t LOPO analysis, do not ask the Oracle.")
    model_with_sel = models.get_model(profile_name, selector=True, lopo=True)
    _, oracle = train.perform_lopo(
        features, metadata, model_with_sel, profile_name, condition, save=save_path
    )

    print("\t oLOPO analysis, ask the Oracle.")
    model_wo_sel = models.get_model(profile_name, selector=False)
    train.perform_lopo(
        features,
        metadata,
        model_wo_sel,
        profile_name,
        condition,
        which_oracle=oracle,
        save=save_path,
    )

    print("\t Stability analysis.")
    train.perform_stability_analysis(
        features, metadata, model, profile_name, condition, save=save_path
    )

    print("Analysis")
    analysis.build_analysis(
        features, metadata, profile_name, condition, "healthy", save_path
    )


if __name__ == "__main__":
    import sys

    _, this_condition, this_profile_name, this_path = sys.argv
    this_path = pathlib.Path(this_path)

    main(
        this_condition,
        profile_name=this_profile_name,
        build_data=False,
        sync=False,
        debug=False,
        ext="jbl",
        path=this_path,
    )
