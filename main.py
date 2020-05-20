#!/usr/bin/env python
# coding: utf-8
import numpy as np
import random
from mlgut import datasets
from mlgut import train
from mlgut import models
import subprocess

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


def main(condition, profile_name, build_data=True, sync=True):
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

    train_interpreter(condition, profile_name)


def train_interpreter(condition, profile_name):
    """[summary]

    Parameters
    ----------
    condition : [type]
        [description]
    profile_name : [type]
        [description]
    """
    features, metadata = datasets.build_condition_dataset(condition, profile_name)
    model = models.get_model(profile_name)

    train.perform_stability_analysis(features, metadata, model, profile_name, condition)
    train.perform_crossproject_analysis(
        features, metadata, model, profile_name, condition
    )

    model_wo_sel = models.get_model(profile_name, selector=False)
    _, oracle = train.perform_lopo(features, metadata, model, profile_name, condition)
    train.perform_lopo(
        features, metadata, model_wo_sel, profile_name, condition, which_oracle=oracle
    )


if __name__ == "__main__":
    condition = "CRC"
    profile_name = "centrifuge"
    main(condition, profile_name)
