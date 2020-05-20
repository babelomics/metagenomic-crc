# coding: utf-8
"""
author: Carlos Loucera
email: carlos.loucera@juntadeandalucia.es

Train functions.
"""
from sklearn.pipeline import Pipeline
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.model_selection import cross_validate, StratifiedShuffleSplit

SCORE_LIST = [
    "roc_auc",
    "average_precision",
    "accuracy",
    "f1",
    "precision",
    "recall",
]


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

    for project_id in metadata.SECONDARY_STUDY_ID.unique():
        stability_results[project_id] = {}

        project_query = metadata.SECONDARY_STUDY_ID == project_id
        condition_query = metadata.DISEASE.isin([condition, control])

        query = project_query & condition_query
        X = features.loc[query, :]
        y = metadata.DISEASE[query]
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

    return stability_results
