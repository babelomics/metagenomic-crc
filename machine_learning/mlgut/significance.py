# coding: utf-8
"""
author: Carlos Loucera
email: carlos.loucera@juntadeandalucia.es

Run signature significance analysis.
"""
import pathlib
import sys

import joblib
import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier
from scipy.stats import iqr
from sklearn import feature_selection as skfs
from sklearn import model_selection as skms
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from mlgut import datasets, models

MODELS = ["LOPO", "oLOPO_withCrossSupport", "oLOPO_withSignature"]
DISEASE_COLUMN_NAME = "DISEASE"
PROJECT_COLUMN_NAME = "SECONDARY_STUDY_ID"

N_CPU = 24


def get_lopo_model(model_name, condition, profile_name, X, y, g, results_path):
    """Retrieve the model.

    Parameters
    ----------
    model_name : str like
        Model name to fill metadata.
    condition : str like
        Disease code.
    profile_name : str like
        Metagenomics profile name.
    X : array like (n_samples, n_features)
        Features.
    y : array like, (n_samples, )
        Target vector relative to X.
    g : array like (n-samples, )
        Groups related to the target.
    results_path : str path like
        Profile-Condition asscoiated path.

    Returns
    -------
    dict
        Valdation results.

    Raises
    ------
    NotImplementedError
        [description]
    """
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
        model = get_olopo_with_support(
            condition, profile_name, "project", X, y, g, results_path
        )
    elif model_name == "oLOPO_withSignature":
        model = get_olopo_with_support(
            condition, profile_name, "signature", X, y, g, results_path
        )
    else:
        raise NotImplementedError()

    return model


def remove_healthy_batch(features, y, g, selector=skfs.SelectFdr()):
    """Noise removal tool based on the healthy condition and the groups. Remove those 
    features that can discriminate between the healthy aong the groups (projects).

    Parameters
    ----------
    features : array like, (n_samples, n_features)
        Metagenomic feature profile.
    y : array lie, (n_samples, )
        Target vector relative to features.
    g : array like, (n_samples, )
        Groups vector related to target.
    selector : skfs.SelectorMixin, optional
        An univariate feature selection sklearn procedure, by default skfs.SelectFdr()

    Returns
    -------
    array like, (n_samples, n_features_filt)
        The features filtered.
    """
    selector = clone(selector)
    healthy_query = y == False
    features_healthy = features.loc[healthy_query, :]
    label = g[healthy_query]

    selector.fit(features_healthy, label)
    support = selector.get_support()

    features_filt = features.loc[:, ~support]

    return features_filt


def remove_iqr(features, threshold=0.0):
    """IQR filtering.

    Parameters
    ----------
    features : array like, (n_samples, n_features)
        Metagenomic feature profile
    threshold : float, optional
        The threshold to remove features based on IQR, by default 0.0

    Returns
    -------
    array like (n_samples, n_features_filt)
        Features filtered.
    """

    support = features.apply(iqr) > threshold

    return features.loc[:, support]


def get_olopo_with_support(condition, profile_name, modus, X, y, g, results_path):
    """[summary]

    Parameters
    ----------
    condition : str like
        Diseae codntioon code.
    profile_name : str like
        Metagenoicsprofile idnentifier.
    modus : str like
        Siganture computation modality.
    X : array like, (n_samples, n_features)
        Features.
    y : array like, (n_samples, )
        Target related to features.
    g : array like, (n_samples, )
        Groups related to samples.
    results_path : str path like
        Condition-Profile save path.

    Returns
    -------
    Pipeline
        The ML pipeline.
    """
    if modus == "signature":
        folder_path = pathlib.Path(results_path)
        columns_to_keep = models.extract_support_from_signature_path(
            condition=condition, profile_name=profile_name, folder_path=folder_path
        )

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


def train_model(model_name, condition, profile_name, X, y, g, results_path):
    """Where we run the actual validation procedure.

    Parameters
    ----------
    model_name : str like
        Model name to fill metadata.
    condition : str like
        Disease code.
    profile_name : str like
        Metagenomics profile name.
    X : array like (n_samples, n_features)
        Features.
    y : array like, (n_samples, )
        Target vector relative to X.
    g : array like (n-samples, )
        Groups related to the target.
    results_path : str path like
        Profile-Condition asscoiated path.

    Returns
    -------
    dict
        Valdation results.
    """
    model = get_lopo_model(model_name, condition, profile_name, X, y, g, results_path)
    lopo_cv = skms.LeaveOneGroupOut().split(X, y, groups=g)
    score, permutation_scores, pvalue = skms.permutation_test_score(
        clone(model), X, y, scoring="roc_auc", cv=lopo_cv, n_permutations=100, n_jobs=-1
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
    """Helper function to stoe the results.

    Parameters
    ----------
    results : dict
        The validation results.
    path : str path like
        Where to store the results.
    """
    condition = results["condition"]
    profile_name = results["profile_name"]
    model_name = results["model_name"]

    fname = f"{condition}_{profile_name}_{model_name}.jbl"
    fpath = path.joinpath(fname)
    joblib.dump(results, fpath)


def train_profile(condition, profile_name, model_name, path):
    """Launch validation procedure for a given profile.

    Parameters
    ----------
    condition : str like
        Disease code.
    profile_name : str like
        Metagenomics profile name.
    model_name : str like
        Name used to dump the model.
    path : str path like
        Where the learned signature was stored. Where to save the results.

    Returns
    -------
    dict
        Results structure.
    """
    features, metadata = datasets.build_condition_dataset(
        condition, profile_name, ext="jbl"
    )

    if profile_name == "OGs":
        features = datasets.filter_egg(features)

    query = metadata[DISEASE_COLUMN_NAME].str.lower().str.contains("crc|healthy")
    features = features.loc[query, :].copy()
    target = (
        ~metadata.loc[query, DISEASE_COLUMN_NAME].str.lower().str.contains("healthy")
    )
    groups = metadata.loc[query, PROJECT_COLUMN_NAME]

    results = train_model(
        model_name, condition, profile_name, features, target, groups, path
    )
    results["condition"] = condition
    save_results(results, path)

    return results


def run_(condition, profile, model_name, path):
    """Launch validation procedure for a given profile.

    Parameters
    ----------
    condition : str like
        Disease code.
    profile_name : str like
        Metagenomics profile name.
    model_name : str like
        Name used to dump the model.
    path : str path like
        Where the learned signature was stored. Where to save the results.

    Raises
    ------
    NotImplementedError
        [description]
    """
    results = train_profile(condition, profile, model_name, path)
    save_results(results, path)


if __name__ == "__main__":
    _, this_condition, this_profile, this_model_name, this_path = sys.argv
    this_path = pathlib.Path(this_path)

    run_(this_condition, this_profile, this_model_name, this_path)
