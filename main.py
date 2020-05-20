#!/usr/bin/env python
# coding: utf-8
import numpy as np
import random
from mlgut import datasets
from mlgut import train
from mlgut import models


SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def build_data_sources():
    metadata = datasets.build_metadata()
    metadata, features_dict = datasets.build_features(metadata)
    datasets.write_metadata(metadata)
    datasets.write_features(features_dict)


def main():
    # build_data_sources()
    condition = "CRC"
    profile_name = "centrifuge"

    features, metadata = datasets.build_condition_dataset(condition, profile_name)
    model = models.get_model(profile_name)

    train.perform_stability_analysis(features, metadata, model, profile_name, condition)
    train.perform_crossproject_analysis(
        features, metadata, model, profile_name, condition
    )


if __name__ == "__main__":
    main()
