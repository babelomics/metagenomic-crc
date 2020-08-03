# coding: utf-8
"""
author: Carlos Loucera
email: carlos.loucera@juntadeandalucia.es

Run LOPO significance anlysis.
"""
import pathlib
import sys

import joblib
import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn import feature_selection as skfs
from sklearn import model_selection as skms
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from mlgut import datasets
from mlgut.models import (
    compute_rbo_mat,
    compute_support_ebm,
    get_cp_support,
    get_lopo_support,
)

MODELS = ["oLOPO_withSignature"]
DISEASE_COLUMN_NAME = "DISEASE"
PROJECT_COLUMN_NAME = "SECONDARY_STUDY_ID"

N_CPU = 24


def get_lopo_model(model_name, profile_name, X, y, g):
    if model_name == "LOPO":
        n_bins = 2 if profile_name == "centrifuge" else 20
        model = Pipeline(
            [
                ("transformer", FunctionTransformer(np.log1p)),
                ("selector", skfs.SelectFdr()),
                (
                    "estimator",
                    ExplainableBoostingClassifier(
                        n_estimators=N_CPU,
                        n_jobs=-1,
                        random_state=42,
                        max_n_bins=n_bins,
                    ),
                ),
            ]
        )
    elif model_name == "oLOPO_withCrossSupport":
        model = get_olopo_with_support(profile_name, "project", X, y, g)
    elif model_name == "oLOPO_withSignature":
        model = get_olopo_with_support(profile_name, "signature", X, y, g)
    else:
        raise NotImplementedError()

    return model


def remove_healthy_batch(features, y, g, selector=skfs.SelectFdr()):
    selector = clone(selector)
    healthy_query = y == False
    features_healthy = features.loc[healthy_query, :]
    label = g[healthy_query]

    selector.fit(features_healthy, label)
    support = selector.get_support()

    features_filt = features.loc[:, ~support]

    return features_filt


def remove_iqr(features, threshold=0.0):
    from scipy.stats import iqr

    support = features.apply(iqr) > threshold

    return features.loc[:, support]


def get_olopo_with_support(profile_name, modus, X, y, g):
    if modus == "signature":
        X = remove_iqr(X)
        X = remove_healthy_batch(X, y, g)
        k = 600 if 600 < (X.shape[1]) else X.shape[1]
        selector = skfs.SelectKBest(k=k).fit(X, y)
        support = selector.get_support()
    else:
        support = []
        for project in g:
            query = g == project
            if project in ["Hannigan", "PRJNA389927"]:
                selector = skfs.SelectFpr()
            else:
                selector = skfs.SelectFdr()
            selector.fit(X.loc[query, :], y.loc[query])
            support.append(selector.get_support())

            support = np.array(support).any(axis=0)

    columns_to_keep = X.columns[support]
    print(columns_to_keep.size)

    trans = ColumnTransformer(
        [("select_columns", "passthrough", columns_to_keep)], remainder="drop",
    )

    n_bins = 2 if profile_name == "centrifuge" else 20

    model = Pipeline(
        [
            ("transformer", FunctionTransformer(np.log1p)),
            (
                "estimator",
                ExplainableBoostingClassifier(
                    n_estimators=N_CPU, n_jobs=-1, random_state=42, max_n_bins=n_bins
                ),
            ),
        ]
    )

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

    print(results)

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


def run_plot(condition, profiles, path):
    pass


def run(condition, profile, mode, path):
    if mode == "train":
        train_profile(condition, profile, path)
    elif mode == "plot":
        run_plot(condition, profile, path)
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    _, condition, profile, mode, path = sys.argv
    path = pathlib.Path(path)

    run(condition, profile, mode, path)
