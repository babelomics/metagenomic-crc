# coding: utf-8
"""
author: Carlos Loucera
email: carlos.loucera@juntadeandalucia.es

Run LOPO significance anlysis.
"""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn import metrics
from sklearn import model_selection as skms

import mlgut.stability as stab
from mlgut import utils
from mlgut import datasets
from mlgut import models
from mlgut.datasets import get_path
from mlgut.models import (
    compute_rbo_mat,
    compute_support_ebm,
    get_cp_support,
    get_lopo_support,
)
import sys

from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from sklearn import feature_selection as skfs
from sklearn.pipeline import Pipeline
import pathlib


MODELS = ["LOPO", "oLOPO_withCrossSupport", "oLOPO_withSignature"]
DISEASE_COLUMN_NAME = "DISEASE"
PROJECT_COLUMN_NAME = "SECONDARY_STUDY_ID"


def get_lopo_model(model_name, profile_name, X, y, g):
    if model_name == "LOPO":
        model = models.get_model(profile=profile_name, selector=True, lopo=True)
    elif model_name == "oLOPO_withCrossSupport":
        model = get_olopo_with_support(profile_name, "project", X, y, g)
    elif model_name == "oLOPO_withSignature":
        model = get_olopo_with_support(profile_name, "signature", X, y, g)
    else:
        raise NotImplementedError()

    return model


def get_olopo_with_support(profile_name, modus, X, y, g):

    support = []

    for project in g:
        query = g == project
        if modus == "signature":
            selector = skfs.SelectKBest(k=100)
        elif modus == "project":
            if project in ["Hannigan", "PRJNA389927"]:
                selector = skfs.SelectFpr()
            else:
                selector = skfs.SelectFdr()
        else:
            raise NotImplementedError()

        selector.fit(X.loc[query, :], y.loc[query])
        support.append(selector.get_support())

    support = np.array(support).any(axis=0)
    columns_to_keep = X.columns[support]

    trans = ColumnTransformer(
        [("select_columns", "passthrough", columns_to_keep)], remainder="drop",
    )

    model = models.get_model(profile=profile_name, selector=False, lopo=True)

    pipe = Pipeline(steps=[("trans", trans), ("model", model)])

    return pipe


def train_model(model_name, profile_name, X, y, g):
    model = get_lopo_model(model_name, profile_name, X, y, g)
    cv = skms.LeaveOneGroupOut().split(X, y, groups=g)
    score, permutation_scores, pvalue = skms.permutation_test_score(
        model, X, y, scoring="roc_auc", cv=cv, n_permutations=100, n_jobs=-1
    )

    results = {
        "model_name": model_name,
        "score": score,
        "permutation_scores": permutation_scores,
        "pvalue": pvalue,
        "profile_name": profile_name,
    }

    return results


def save_results(results, path):
    condition = results["condition"]
    profile_name = results["profile_name"]
    model_name = results["model_name"]

    fname = f"{condition}_{profile_name}_{model_name}.jbl"
    fpath = path.joinpath(fname)
    joblib.dump(results, fpath)


def train_profile(condition, profile_name, path):
    features, metadata = datasets.build_condition_dataset(
        condition, profile_name, ext="jbl"
    )

    if profile_name == "OGs":
        features = datasets.filter_egg(features)

    query = metadata[DISEASE_COLUMN_NAME].str.lower().str.contains("crc|healthy")
    X = features.loc[query, :].copy()
    y = ~metadata.loc[query, DISEASE_COLUMN_NAME].str.lower().str.contains("healthy")
    g = metadata.loc[query, PROJECT_COLUMN_NAME]

    for model_name in MODELS:
        results = train_model(model_name, profile_name, X, y, g)
        results["condition"] = condition
        save_results(results, path)


def run_train(condition, profiles, path):
    for profile_name in profiles:
        train_profile(condition, profile_name, path)


def run_plot(condition, profiles, path):
    pass


def run(condition, profiles, mode, path):
    if mode == "train":
        run_train(condition, profiles, path)
    elif mode == "plot":
        run_plot(condition, profiles, path)
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    _, condition, profiles, mode, path = sys.argv
    path = pathlib.Path(path)

    run(condition, profiles, mode, path)
