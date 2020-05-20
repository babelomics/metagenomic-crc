# coding: utf-8
"""
author: Carlos Loucera
email: carlos.loucera@juntadeandalucia.es

ML models module.
"""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer
from sklearn.feature_selection import SelectFpr
from interpret.glassbox import ExplainableBoostingClassifier


def get_model(profile: str, selector=True) -> Pipeline:
    """[summary]

    Parameters
    ----------
    profile : str
        [description]
    selector : bool, optional
        [description], by default True

    Returns
    -------
    Pipeline
        [description]

    Raises
    ------
    NotImplementedError
        [description]
    """
    if profile.lower() in ["taxo", "taxonomic", "centrifuge"]:
        model = get_taxonomic_model()
    else:
        raise NotImplementedError

    return model


def get_taxonomic_model(selector=True) -> Pipeline:
    """[summary]

    Parameters
    ----------
    selector : bool, optional
        [description], by default True

    Returns
    -------
    Pipeline
        [description]
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
                        n_estimators=32, n_jobs=-1, random_state=42
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
                        n_estimators=32, n_jobs=-1, random_state=42
                    ),
                ),
            ]
        )

    return model
