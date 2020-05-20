#!/usr/bin/env python

from mlgut import datasets


def build_data_sources():
    metadata = datasets.build_metadata()
    metadata, features_dict = datasets.build_features(metadata)
    datasets.write_metadata(metadata)
    datasets.write_features(features_dict)


if __name__ == "__main__":
    build_data_sources()