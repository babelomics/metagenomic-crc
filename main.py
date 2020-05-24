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


def build_data_sources():
    """[summary]
    """
    metadata = datasets.build_metadata()
    metadata, features_dict = datasets.build_features(metadata)
    datasets.write_metadata(metadata)
    datasets.write_features(features_dict)


def main(condition, profile_name, build_data=True, sync=True, debug=True):
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
        subprocess.run(["sh", "mlgut/sync_data.sh"])
    if build_data:
        build_data_sources()
    if debug:
        filter_warnings()

    train_interpreter(condition, profile_name)


def filter_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)


def train_interpreter(condition, profile_name):
    """[summary]

    Parameters
    ----------
    condition : [type]
        [description]
    profile_name : [type]
        [description]
    """
    print(f"Building datasets for {condition} condition and profile {profile_name}")
    features, metadata = datasets.build_condition_dataset(condition, profile_name)
    model = models.get_model(profile_name)
    print(model)

    print("\t Stability analysis.")
    train.perform_stability_analysis(features, metadata, model, profile_name, condition)

    print("\t Cross-project analysis.")
    train.perform_crossproject_analysis(
        features, metadata, model, profile_name, condition
    )

    print("\t LOPO analysis, do not ask the Oracle.")
    model_wo_sel = models.get_model(profile_name, selector=False)
    _, oracle = train.perform_lopo(features, metadata, model, profile_name, condition)

    print("\t LOPO analysis, ask the Oracle.")
    train.perform_lopo(
        features, metadata, model_wo_sel, profile_name, condition, which_oracle=oracle
    )

    path = utils.get_path("results")
    print("Analysis")
    analysis.build_analysis(
        features, metadata, profile_name, condition, "healthy", path
    )


if __name__ == "__main__":
    condition = "CRC"
    profile_name = "KEGG_KOs"
    main(condition, profile_name, build_data=False, sync=False)
