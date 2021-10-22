# coding: utf-8
"""
author: Carlos Loucera
email: carlos.loucera@juntadeandalucia.es

ML models module.
"""
import pathlib

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.feature_selection import SelectFdr, SelectFpr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer

N_ESTIMATORS = 24


def get_model(profile: str, selector=True, lopo=False) -> Pipeline:
    """Get an unfitted estimator as a sklearn pipeline for each metagenomic profile.

    Parameters
    ----------
    profile : str
        Metagenomics profile name (one of OGs: eggNog, KEGG_KOs: kegg, centrifuge: taxo)
    selector : bool, optional
        Use FS as a layer, by default True

    Returns
    -------
    Pipeline
        Estimator as a reproduble pipeline

    Raises
    ------
    NotImplementedError
        Profile not included yet
    """
    if profile.lower() in ["taxo", "taxonomic", "centrifuge"]:
        model = get_taxonomic_model(selector, lopo)
    elif "kegg" in profile.lower():
        model = get_kegg_model(selector, lopo)
    elif profile.lower() in ["ogs", "egg"]:
        model = get_ogs_model(selector, lopo)
    else:
        raise NotImplementedError

    return model


def get_taxonomic_model(selector=True, lopo=False) -> Pipeline:
    """Get an unfitted estimator as a sklearn pipeline for taxonomic profile.

    Parameters
    ----------
    selector : bool, optional
        Use FS as a layer, by default True

    Returns
    -------
    Pipeline
        Estimator as a reproduble pipeline
    """
    if selector:
        model = Pipeline(
            [
                ("transformer", FunctionTransformer(np.log1p)),
                ("discretizer", KBinsDiscretizer(n_bins=4, encode="ordinal")),
                ("selector", SelectFpr()),
                (
                    "estimator",
                    ExplainableBoostingClassifier(
                        n_estimators=N_ESTIMATORS, n_jobs=-1, random_state=42
                    ),
                ),
            ]
        )

        if lopo:
            model = Pipeline(
                [
                    ("transformer", FunctionTransformer(np.log1p)),
                    ("discretizer", KBinsDiscretizer(n_bins=4, encode="ordinal")),
                    ("selector", SelectFdr()),
                    (
                        "estimator",
                        ExplainableBoostingClassifier(
                            n_estimators=N_ESTIMATORS, n_jobs=-1, random_state=42
                        ),
                    ),
                ]
            )

    else:
        model = Pipeline(
            [
                ("transformer", FunctionTransformer(np.log1p)),
                ("discretizer", KBinsDiscretizer(n_bins=4, encode="ordinal")),
                (
                    "estimator",
                    ExplainableBoostingClassifier(
                        n_estimators=N_ESTIMATORS, n_jobs=-1, random_state=42
                    ),
                ),
            ]
        )

    return model


def get_kegg_model(selector=True, lopo=False) -> Pipeline:
    """Get an unfitted estimator as a sklearn pipeline for KEGG KO profile.

    Parameters
    ----------
    selector : bool, optional
        Use FS as a layer, by default True

    Returns
    -------
    Pipeline
        Estimator as a reproduble pipeline
    """
    if selector:
        model = Pipeline(
            [
                ("transformer", FunctionTransformer(np.log1p)),
                ("discretizer", KBinsDiscretizer(n_bins=4, encode="ordinal")),
                ("selector", SelectFpr()),
                (
                    "estimator",
                    ExplainableBoostingClassifier(
                        n_estimators=N_ESTIMATORS, n_jobs=-1, random_state=42
                    ),
                ),
            ]
        )

        if lopo:
            model = Pipeline(
                [
                    ("transformer", FunctionTransformer(np.log1p)),
                    ("discretizer", KBinsDiscretizer(n_bins=4, encode="ordinal")),
                    ("selector", SelectFdr()),
                    (
                        "estimator",
                        ExplainableBoostingClassifier(
                            n_estimators=N_ESTIMATORS, n_jobs=-1, random_state=42
                        ),
                    ),
                ]
            )

    else:
        model = Pipeline(
            [
                ("transformer", FunctionTransformer(np.log1p)),
                ("discretizer", KBinsDiscretizer(n_bins=4, encode="ordinal")),
                (
                    "estimator",
                    ExplainableBoostingClassifier(
                        n_estimators=N_ESTIMATORS, n_jobs=-1, random_state=42
                    ),
                ),
            ]
        )

    return model


def get_ogs_model(selector=True, lopo=False, k=20) -> Pipeline:
    """Get an unfitted estimator as a sklearn pipeline for eggNog profile.

    Parameters
    ----------
    selector : bool, optional
        Use FS as a layer, by default True

    Returns
    -------
    Pipeline
        Estimator as a reproduble pipeline
    """
    if selector:
        model = Pipeline(
            [
                ("transformer", FunctionTransformer(np.log1p)),
                ("selector", SelectFpr()),
                ("discretizer", KBinsDiscretizer(n_bins=k, encode="ordinal")),
                (
                    "estimator",
                    ExplainableBoostingClassifier(
                        n_estimators=N_ESTIMATORS, n_jobs=-1, random_state=42
                    ),
                ),
            ]
        )

        if lopo:
            model = Pipeline(
                [
                    ("transformer", FunctionTransformer(np.log1p)),
                    ("selector", SelectFdr()),
                    ("discretizer", KBinsDiscretizer(n_bins=k, encode="ordinal")),
                    (
                        "estimator",
                        ExplainableBoostingClassifier(
                            n_estimators=N_ESTIMATORS, n_jobs=-1, random_state=42
                        ),
                    ),
                ]
            )

    else:
        model = Pipeline(
            [
                ("transformer", FunctionTransformer(np.log1p)),
                ("discretizer", KBinsDiscretizer(n_bins=k, encode="ordinal")),
                (
                    "estimator",
                    ExplainableBoostingClassifier(
                        n_estimators=N_ESTIMATORS, n_jobs=-1, random_state=42
                    ),
                ),
            ]
        )

    return model


def compute_support_ebm(model: Pipeline) -> (np.ndarray, np.ndarray):
    """Get the learned relevances.

    Parameters
    ----------
    model : Pipeline
        Fitted estimator.

    Returns
    -------
    ndarray (n_features, ), ndarray (n_features, )
        Support and relevances learned.
    """
    # TODO: check if trained
    ebm = model["estimator"]
    ebm_global = ebm.explain_global()
    data = ebm_global.data()

    if "selector" in model.named_steps.keys():
        support = model["selector"].get_support()
    else:
        support = np.repeat(True, len(data["scores"]))

    coefs = np.zeros(support.size)
    coefs[support] = np.array(data["scores"])

    support = support * 1

    return support, coefs


def get_lopo_support(cv_results, columns):
    """Extract the support learned across the LOPO analysis for a given profile.

    Parameters
    ----------
    cv_results : sklearn CV object
        The model evaluated across the LOPO
    columns : list like (n_features, )
        Feature names.
    
    Returns
    -------
    DataFrame (n_features, n_projects)
        Support for the combined projects. So project_i means features learned using all
        projects except i.
    """

    support = [
        pd.Series(compute_support_ebm(est)[1], index=columns)
        for est in cv_results["estimator"]
    ]

    return combine_support(support)


def get_cp_support(results, columns):
    """Combine the learned support for the cross project analysis (for a given profile).

    Parameters
    ----------
    results : ndarray (n_features, )
        Support learned for one project.
    columns : list like[str] (n_features, )
        Feature names.

    Returns
    -------
    DataFrame (n_features, n_projects)
        Frame with the support learned for each project
    """

    support = [
        pd.Series(
            compute_support_ebm(results[project]["model"])[1],
            index=columns,
            name=project,
        )
        for project in results.keys()
    ]

    return combine_support(support)


def combine_support(support: list):
    """Build the final signature from the learned relevances across the projects.

    Parameters
    ----------
    support : list like
        A list containing the different supports learned.

    Returns
    -------
    DataFrame (n_features, n_projects), DataFrame (n_features, )
        The learned support (raw) and the combined signature.
    """
    support = pd.concat(support, axis=1)
    n_projects = support.shape[1]
    cross_dataset_support = n_projects - (support != 0.0).sum(axis=1) + 1
    support_merged = support / support.max()
    support_merged = support_merged.sum(axis=1)
    support_merged /= cross_dataset_support
    support_merged = support_merged.sort_values(ascending=False)

    return support, support_merged


def extract_support_from_signature(support_frame: pd.DataFrame) -> np.ndarray:
    """Given a profile siganture, extract the support.

    Parameters
    ----------
    support_frame : pd.DataFrame (n_features, )
        The model relevances.

    Returns
    -------
    np.ndarray (n_features_used, )
        Support as feature names.
    """
    return support_frame[support_frame > 0.0].index.astype(str)


def extract_support_from_signature_path(condition, profile_name, folder_path):
    """Load a signature stored with a support-based function and extract the support.

    Parameters
    ----------
    condition : str like
        Disease code name.
    profile_name : str like
        Metagenomics profile name.
    folder_path : str like
        Folder where the signature was dumped.

    Returns
    -------
    [type]
        [description]
    """
    folder_path = pathlib.Path(folder_path)
    best_path = folder_path.joinpath(f"{condition}_{profile_name}_cp_support_merge.tsv")

    support_frame = pd.read_csv(best_path, sep="\t", index_col=0).iloc[:, 0]
    columns = extract_support_from_signature(support_frame)

    return columns
