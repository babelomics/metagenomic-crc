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
from scipy.spatial.distance import pdist
from mlgut.rbo import rbo_dict


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


def compute_rbo_mat(rank_mat_filt, p=0.999, filt=True):
    if filt:
        # safe filter to speed up the computation
        rank_mat_filt = rank_mat_filt.loc[rank_mat_filt.any(axis=1), :]

    col_dict_list = np.array(
        [rank_mat_filt[col].to_dict() for col in rank_mat_filt.columns]
    ).reshape(-1, 1)
    distmat = pdist(col_dict_list, lambda x, y: 1 - rbo_dict(x[0], y[0], p=p)[-1])

    return distmat


def compute_support_ebm(model, quantile=None):
    #TODO: check if trained
    support = model["selector"].get_support()
    
    coefs = np.zeros(support.size)
    
    ebm = model["estimator"]
    ebm_global = ebm.explain_global()
    data = ebm_global.data()
    coefs[support] = np.array(data["scores"])
    
    support = support * 1
    
    return support, coefs