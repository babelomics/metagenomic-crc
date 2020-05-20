# coding: utf-8
"""
author: Carlos Loucera
email: carlos.loucera@juntadeandalucia.es

Train functions.
"""
from sklearn.pipeline import Pipeline
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.model_selection import (
    cross_validate,
    StratifiedShuffleSplit,
    LeaveOneGroupOut,
    RepeatedStratifiedKFold,
)
import joblib
from typing import Tuple, Iterable


SCORE_LIST = [
    "roc_auc",
    "average_precision",
    "accuracy",
    "f1",
    "precision",
    "recall",
]

DISEASE_COLUMN_NAME = "DISEASE"
PROJECT_COLUMN_NAME = "SECONDARY_STUDY_ID"


def perform_stability_analysis(
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    model: Pipeline,
    profile: str,
    condition: str,
    control="healthy",
    n_splits=100,
    frac_samples=0.7,
    save=True,
) -> dict:
    """[summary]

    Parameters
    ----------
    features : pd.DataFrame
        [description]
    metadata : pd.DataFrame
        [description]
    model : Pipeline
        [description]
    profile : str
        [description]
    condition : str
        [description]
    control : str, optional
        [description], by default "healthy"
    n_splits : int, optional
        [description], by default 100
    frac_samples : float, optional
        [description], by default 0.7
    save : bool, optional
        [description], by default True

    Returns
    -------
    dict
        [description]
    """

    stability_results = {}

    for project_id in metadata[PROJECT_COLUMN_NAME].unique():
        stability_results[project_id] = {}

        project_query = metadata[PROJECT_COLUMN_NAME] == project_id
        condition_query = metadata[DISEASE_COLUMN_NAME].isin([condition, control])

        query = project_query & condition_query
        X = features.loc[query, :]
        y = metadata[DISEASE_COLUMN_NAME][query]
        y = y == condition

        n_examples = int(frac_samples * X.shape[0])
        stability_cv = StratifiedShuffleSplit(
            n_splits=n_splits, train_size=n_examples, random_state=0
        )

        res = cross_validate(
            model,
            X,
            y,
            cv=stability_cv,
            scoring=SCORE_LIST,
            return_estimator=True,
            return_train_score=True,
            n_jobs=-1,
        )
        stability_results[project_id] = res

    if save:
        joblib.dump(stability_results, f"{condition}_{profile}_stability.jbl")

    return stability_results


def perform_crossproject_analysis(
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    model: Pipeline,
    profile: str,
    condition: str,
    control="healthy",
    save=True,
) -> dict:

    query = metadata[DISEASE_COLUMN_NAME].isin([control, condition])
    X = features.loc[query, :]
    y = metadata.loc[query, DISEASE_COLUMN_NAME] == condition
    g = metadata.loc[query, PROJECT_COLUMN_NAME]

    results = {}
    for project_id in metadata[PROJECT_COLUMN_NAME].unique():
        results[project_id] = {}

        project_query = metadata[PROJECT_COLUMN_NAME] == project_id
        condition_query = metadata[DISEASE_COLUMN_NAME].isin([condition, control])
        query = project_query & condition_query

        X = features.loc[query, :]
        y = metadata[DISEASE_COLUMN_NAME][query]
        y = y == condition

        query_out = (~project_query) & condition_query
        X_out = features.loc[query_out, :]
        y_out = metadata[DISEASE_COLUMN_NAME][query_out]
        y_out = y_out == condition

        model.fit(X, y)
        results[project_id]["model"] = model
        outer_score = model.predict_proba(X_out)[:, 1]
        outer_score = pd.Series(outer_score, index=X_out.index)
        outer_score.name = "decission"

        results[project_id]["outer"] = pd.concat(
            (metadata.loc[query_out, :], outer_score), axis=1
        )

        inner_project_cv = RepeatedStratifiedKFold(
            n_splits=10, n_repeats=20, random_state=0
        )

        rescv = cross_validate(
            model,
            X,
            y,
            cv=inner_project_cv,
            scoring=SCORE_LIST,
            return_estimator=True,
            return_train_score=True,
            n_jobs=1,
        )

        results[project_id]["cv"] = rescv

    if save:
        joblib.dump(results, f"{condition}_{profile}_cross_project.jbl")

    return results


def perform_lopo(
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    model: Pipeline,
    profile: str,
    condition: str,
    control="healthy",
    save=True,
    which_oracle=None,
) -> Tuple[dict, pd.DataFrame]:
    """[summary]

    Parameters
    ----------
    features : pd.DataFrame
        [description]
    metadata : pd.DataFrame
        [description]
    model : Pipeline
        [description]
    profile : str
        [description]
    condition : str
        [description]
    control : str, optional
        [description], by default "healthy"
    save : bool, optional
        [description], by default True
    which_oracle : [type], optional
        [description], by default None

    Returns
    -------
    Tuple[dict, pd.DataFrame]
        [description]
    """

    oracle = which_oracle

    if which_oracle in [None, False]:
        results, oracle = perform_lopo_wo_oracle(
            features, metadata, model, profile, condition, control
        )
    else:
        results, oracle = perform_lopo_with_oracle(
            features, metadata, model, profile, condition, which_oracle, control
        )

    return results, oracle


def perform_lopo_wo_oracle(
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    model: Pipeline,
    profile: str,
    condition: str,
    control="healthy",
    save=True,
) -> Tuple[dict, pd.DataFrame]:
    """[summary]

    Parameters
    ----------
    features : pd.DataFrame
        [description]
    metadata : pd.DataFrame
        [description]
    model : Pipeline
        [description]
    profile : str
        [description]
    condition : str
        [description]
    control : str, optional
        [description], by default "healthy"
    save : bool, optional
        [description], by default True

    Returns
    -------
    Tuple[dict, pd.DataFrame]
        [description]
    """

    query = metadata[DISEASE_COLUMN_NAME].isin([control, condition])
    X = features.loc[query, :]
    y = metadata.loc[query, DISEASE_COLUMN_NAME] == condition
    g = metadata.loc[query, PROJECT_COLUMN_NAME]

    splitter = LeaveOneGroupOut().split(X, y, groups=g)

    results = cross_validate(
        model,
        X,
        y,
        cv=splitter,
        scoring=SCORE_LIST,
        return_estimator=True,
        return_train_score=True,
        n_jobs=-1,
    )

    if save:
        joblib.dump(results, f"{condition}_{profile}_lopo.jbl")

    support_matrix = [
        pd.Series(model_["selector"].get_support(), index=X.columns)
        for model_ in results["estimator"]
    ]

    return results, support_matrix


def perform_lopo_with_oracle(
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    model: Pipeline,
    profile: str,
    condition: str,
    oracle: pd.DataFrame,
    control="healthy",
    save=True,
) -> dict:
    """[summary]

    Parameters
    ----------
    features : pd.DataFrame
        [description]
    metadata : pd.DataFrame
        [description]
    model : Pipeline
        [description]
    profile : str
        [description]
    condition : str
        [description]
    oracle : pd.DataFrame
        [description]
    control : str, optional
        [description], by default "healthy"
    save : bool, optional
        [description], by default True

    Returns
    -------
    dict
        [description]
    """

    query = metadata[DISEASE_COLUMN_NAME].isin([control, condition])
    y = metadata.loc[query, DISEASE_COLUMN_NAME] == condition
    g = metadata.loc[query, PROJECT_COLUMN_NAME]

    # Ask the oracle
    results = {}
    top = pd.concat(oracle, axis=1).sum(axis=1)
    top = top.sort_values(ascending=False)

    for i in range(1, g.unique().size + 1):
        # TODO perform column selection by means of ColumnTransformer
        results[i] = {}
        query_cols = top[top >= i].index.unique()
        X = features.loc[query, query_cols]

        splitter = LeaveOneGroupOut().split(X, y, groups=g)
        model

        cv = cross_validate(
            model,
            X,
            y,
            cv=splitter,
            scoring=SCORE_LIST,
            return_estimator=True,
            return_train_score=True,
            n_jobs=1,
        )
        results[i]["cv"] = cv
        results[i]["columns"] = query_cols

    return results