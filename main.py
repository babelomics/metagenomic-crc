#!/usr/bin/env python
# coding: utf-8
import numpy as np
import random
from mlgut import datasets
from mlgut import train
from mlgut import models
from mlgut import analysis
from mlgut import utils
import subprocess
import warnings
from sklearn.exceptions import ConvergenceWarning


SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def build_data_sources(profiles=["KEGG_KOs", "centrifuge", "OGs"], ext="jbl"):
    """[summary]
    """
    metadata = datasets.build_metadata()
    metadata, features_dict = datasets.build_features(metadata, profiles)
    datasets.write_metadata(metadata, ext=ext)
    datasets.write_features(features_dict, ext=ext)


def main(condition, profile_name, build_data=True, sync=True, debug=True, ext="jbl"):
    """[summary]

    Parameters
    ----------
    condition : [type]
        [description]
    profile_name : [type]
        [description]
    build_data : bool, optional
        [description], by default True
    sync : bool, optional
        [description], by default True
    """
    if sync:
        print("Sync data.")
        subprocess.run(["sh", "mlgut/sync_data.sh"])
    if build_data:
        print("Build Data sources.")
        build_data_sources(ext=ext)
    if not debug:
        filter_warnings()
    if profile_name is not None:
        train_interpreter(condition, profile_name, ext=ext)


def filter_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)


def train_interpreter(condition, profile_name, ext):
    """[summary]

    Parameters
    ----------
    condition : [type]
        [description]
    profile_name : [type]
        [description]
    """
    import pandas as pd

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
        features, metadata, model, profile_name, condition
    )

    print("\t LOPO analysis, do not ask the Oracle.")
    model_with_sel = models.get_model(profile_name, selector=True, lopo=True)
    _, oracle = train.perform_lopo(
        features, metadata, model_with_sel, profile_name, condition
    )

    print("\t LOPO analysis, ask the Oracle.")
    model_wo_sel = models.get_model(profile_name, selector=False)
    train.perform_lopo(
        features, metadata, model_wo_sel, profile_name, condition, which_oracle=oracle
    )

    print("\t Stability analysis.")
    train.perform_stability_analysis(features, metadata, model, profile_name, condition)

    path = utils.get_path("results")
    print("Analysis")
    analysis.build_analysis(
        features, metadata, profile_name, condition, "healthy", path
    )


if __name__ == "__main__":
    condition = "CRC"
    profile_name = "OGs"
    main(
        condition,
        profile_name=None,
        build_data=True,
        sync=False,
        debug=False,
        ext="jbl",
    )
