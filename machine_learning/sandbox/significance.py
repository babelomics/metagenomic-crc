# coding: utf-8
"""
author: Carlos Loucera
email: carlos.loucera@juntadeandalucia.es

Run LOPO significance analysis.
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

MODELS = ["LOPO", "oLOPO_withCrossSupport", "oLOPO_withSignature"]
DISEASE_COLUMN_NAME = "DISEASE"
PROJECT_COLUMN_NAME = "SECONDARY_STUDY_ID"

N_CPU = 24


def get_lopo_model(model_name, profile_name, X, y, g):
    if model_name == "LOPO":
        n_bins = 2 if profile_name == "centrifuge" else 20
        model = Pipeline(
            [
                ("selector", skfs.SelectKBest(k=600)),
                ("transformer", FunctionTransformer(np.log1p)),
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
        if profile_name != "centrifuge":
            X = remove_iqr(X)
            X = remove_healthy_batch(X, y, g)
        k = 600 if 600 < (X.shape[1]) else X.shape[1]
        selector = skfs.SelectKBest(k=k).fit(X, y)
        support = selector.get_support()
    else:
        support = []
        for project in g:
            query = g != project
            # if project in ["Hannigan", "PRJNA389927"]:
            #     selector = skfs.SelectFpr()
            # else:
            #     selector = skfs.SelectFdr()
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
        clone(model), X, y, scoring="roc_auc", cv=cv, n_permutations=100, n_jobs=-1
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


def train_profile(condition, profile_name, model_name, path):
    features, metadata = datasets.build_condition_dataset(
        condition, profile_name, ext="jbl"
    )

    if profile_name == "OGs":
        features = datasets.filter_egg(features)

    query = metadata[DISEASE_COLUMN_NAME].str.lower().str.contains("crc|healthy")
    X = features.loc[query, :].copy()
    y = ~metadata.loc[query, DISEASE_COLUMN_NAME].str.lower().str.contains("healthy")
    g = metadata.loc[query, PROJECT_COLUMN_NAME]

    results = train_model(model_name, profile_name, X, y, g)
    results["condition"] = condition
    save_results(results, path)

    return results


def plot_significance(results, path, modus="signature", n_classes=2):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    sns.set(context="poster", style="white", font_scale=0.9)

    colors = ["b", "C1", "g"]

    fig, ax = plt.subplots(constrained_layout=True, figsize=(16, 9))

    for i, results_i in enumerate(results):
        color_i = colors[i]
        profile_name = results_i["profile_name"]
        if "centri" in profile_name.lower():
            profile_name = "Taxo"
        elif "ogs" in profile_name.lower():
            profile_name = "eNog"
        elif "kegg" in profile_name.lower():
            profile_name = "Kegg"
        pvalue = results_i["pvalue"]
        permutation_scores = results_i["permutation_scores"]
        score = results_i["score"]

        # plt.hist(permutation_scores, 20, label='Permutation scores', edgecolor='black')
        if i == 0:
            sns.kdeplot(
                pd.Series(permutation_scores, name="Permutation Scores"), color="k"
            )
        ylim = [0, 15]
        plt.plot(
            2 * [score],
            ylim,
            f"--{color_i}",
            linewidth=3,
            label=f"{profile_name} score {score:.3f} ({pvalue:.2e})",
        )

    plt.plot(2 * [1.0 / n_classes], ylim, "--k", linewidth=3, label="Luck")

    plt.ylim(ylim)

    plt.xlabel("AUROC score")
    # plt.tight_layout()
    plt.legend(loc="upper left", bbox_to_anchor=(0.27, 1))

    sns.despine()

    for ext in ["pdf", "svg", "png"]:
        fname = f"{condition}_{modus}_permutation_analysis.{ext}"
        fpath = path.joinpath(fname)
        plt.savefig(fpath, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.show()
    plt.close()


def run(condition, profile, mode, model_name, path):
    if mode == "train":
        results = train_profile(condition, profile, model_name, path)
        plot_significance(results, path)
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    _, condition, profile, mode, model_name, path = sys.argv
    path = pathlib.Path(path)

    run(condition, profile, mode, model_name, path)
