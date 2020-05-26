# coding: utf-8
"""
author: Carlos Loucera
email: carlos.loucera@juntadeandalucia.es

ML models module.
"""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer
from sklearn.feature_selection import SelectFpr, SelectFdr
from interpret.glassbox import ExplainableBoostingClassifier
from scipy.spatial.distance import pdist
from mlgut.rbo import rbo_dict
import pandas as pd
from sklearn import metrics


def get_model(profile: str, selector=True, lopo=False) -> Pipeline:
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
        model = get_taxonomic_model(selector, lopo)
    elif "kegg" in profile.lower():
        model = get_kegg_model(selector, lopo)
    else:
        raise NotImplementedError

    return model


def get_taxonomic_model(selector=True, lopo=False) -> Pipeline:
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

        if lopo:
            model = Pipeline(
                [
                    ("transformer", FunctionTransformer(np.log1p)),
                    ("discretizer", KBinsDiscretizer(n_bins=4, encode="ordinal")),
                    ("selector", SelectFdr()),
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


def get_kegg_model(selector=True, lopo=False) -> Pipeline:
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

        if lopo:
            model = Pipeline(
                [
                    ("transformer", FunctionTransformer(np.log1p)),
                    ("discretizer", KBinsDiscretizer(n_bins=4, encode="ordinal")),
                    ("selector", SelectFdr()),
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


# def compute_rbo_mat(rank_mat_filt, p=0.999, filt=True):
#     if filt:
#         # safe filter to speed up the computation
#         rank_mat_filt = rank_mat_filt.loc[rank_mat_filt.any(axis=1), :]

#     col_dict_list = np.array(
#         [rank_mat_filt[col].to_dict() for col in rank_mat_filt.columns]
#     ).reshape(-1, 1)
#     distmat = pdist(col_dict_list, lambda x, y: 1 - rbo_dict(x[0], y[0], p=p)[-1])

#     return distmat


def rbo_dist(x, y, p=0.999):
    x_dict = pd.Series(x).to_dict()
    y_dict = pd.Series(y).to_dict()

    return 1 - rbo_dict(x_dict, y_dict, p=p)[-1]


def compute_rbo_mat(rank_mat, p=0.999, filt=True):
    if filt:
        # safe filter to speed up the computation
        rank_mat_filt = rank_mat.loc[rank_mat.any(axis=1), :]
    else:
        rank_mat_filt = rank_mat

    dmat = metrics.pairwise_distances(X=rank_mat_filt, n_jobs=-1, metric=rbo_dist, p=p)

    return dmat


def compute_support_ebm(model: Pipeline, quantile=None):
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

    support = [
        pd.Series(compute_support_ebm(est)[1], index=columns)
        for est in cv_results["estimator"]
    ]

    return combine_support(support)


def get_cp_support(results, columns):

    support = [
        pd.Series(
            compute_support_ebm(results[project]["model"])[1],
            index=columns,
            name=project,
        )
        for project in results.keys()
    ]

    return combine_support(support)


def combine_support(support):
    support = pd.concat(support, axis=1)
    n_projects = support.shape[1]
    cross_dataset_support = n_projects - (support != 0.0).sum(axis=1) + 1
    support_merged = support / support.max()
    support_merged = support_merged.sum(axis=1)
    support_merged /= cross_dataset_support
    support_merged = support_merged.sort_values(ascending=False)

    return support, support_merged


# class ItemSelector(BaseEstimator, TransformerMixin):
#     """For data grouped by feature, select subset of data at a provided key.

#     The data is expected to be stored in a 2D data structure, where the first
#     index is over features and the second is over samples.  i.e.

#     >> len(data[key]) == n_samples

#     Please note that this is the opposite convention to scikit-learn feature
#     matrixes (where the first index corresponds to sample).

#     ItemSelector only requires that the collection implement getitem
#     (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
#     DataFrame, numpy record array, etc.

#     >> data = {'a': [1, 5, 2, 5, 2, 8],
#                'b': [9, 4, 1, 4, 1, 3]}
#     >> ds = ItemSelector(key='a')
#     >> data['a'] == ds.transform(data)

#     ItemSelector is not designed to handle data grouped by sample.  (e.g. a
#     list of dicts).  If your data is structured this way, consider a
#     transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

#     Parameters
#     ----------
#     key : hashable, required
#         The key corresponding to the desired value in a mappable.
#     """

#     def __init__(self, key):
#         self.key = key

#     def fit(self, x, y=None):
#         return self

#     def transform(self, data_dict):
#         return data_dict[self.key]


# def get_combined_model(profile_list):

#     transformer_list = [
#         ("name", Pipeline([("selector", ItemSelector(key=name))]))
#         for name in profile_list
#     ]

#     transformer_list = []

#     for name in profile_list:
#         if name.lower() == "centrifuge":
#             pipe = (
#                 "name",
#                 Pipeline(
#                     [
#                         ("item", ItemSelector(key=name)),
#                         ("disretizer", KBinsDiscretizer(n_bins=4, encode="ordinal")),
#                         ("selector", SelectFpr()),
#                     ]
#                 ),
#             )

#             transformer_list.append(pipe)

#     model = Pipeline(
#         [
#             # Use FeatureUnion to combine the features from kegg and aro
#             ("union", FeatureUnion(transformer_list)),
#             # component-wise transformation
#             ("transformer", QuantileTransformer(random_state=42)),
#             (
#                 "pca",
#                 decomposition.KernelPCA(
#                     kernel="linear", n_components=100, random_state=42
#                 ),
#             ),
#             #     ('gp', gpf)
#             #     # Use a Linear SVM classifier on the combined features
#             ("svm", SVC(kernel="linear", probability=True, random_state=42)),
#         ]
#     )

#     return model
