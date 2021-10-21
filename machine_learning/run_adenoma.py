# coding: utf-8
"""
author: Carlos Loucera
email: carlos.loucera@juntadeandalucia.es

Adenoma analysis.
"""

import pathlib

import joblib
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from scipy import stats
from scipy.stats import linregress
from sklearn.model_selection import train_test_split

from mlgut import datasets, models, pystab

PROJECT_NAMES_DICT = {
    "PRJNA389927": "Hannigan",
    "PRJEB12449": "Vogtmann",
    "PRJEB6070": "Zeller",
    "PRJEB7774": "Feng",
    "PRJEB10878": "Yu",
    "PRJNA447983": "Thomas0",
    "PRJEB27928": "Thomas1",
}

PROJECT_ORDER = sorted(PROJECT_NAMES_DICT.values())
DISEASE_COLUMN_NAME = "DISEASE"
PROJECT_COLUMN_NAME = "SECONDARY_STUDY_ID"

EXTENSIONS = ["pdf", "png", "svg"]


def main(condition, profile_name, results_path):
    features, metadata = datasets.build_condition_dataset(
        condition, profile_name, ext="jbl"
    )

    _, metadata_adenoma = datasets.build_condition_dataset(
        "adenoma", profile_name, ext="jbl"
    )

    if "ogs" in profile_name.lower():
        features = datasets.filter_egg(features)

    metadata[PROJECT_COLUMN_NAME] = metadata[PROJECT_COLUMN_NAME].replace(
        PROJECT_NAMES_DICT
    )

    metadata_adenoma[PROJECT_COLUMN_NAME] = metadata_adenoma[
        PROJECT_COLUMN_NAME
    ].replace(PROJECT_NAMES_DICT)

    folder_path = pathlib.Path(results_path)
    best_path = folder_path.joinpath(f"{condition}_{profile_name}_cp_support_merge.tsv")

    d = pd.read_csv(best_path, sep="\t", index_col=0).iloc[:, 0]
    columns = d[d > 0.0].index.astype(str)

    features = features.apply(np.log1p)

    query = metadata.DISEASE.isin([condition, "healthy"])
    disease_train = metadata.DISEASE[query]
    y_train = metadata.DISEASE[query] == condition
    X_train = features.loc[query, columns]

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, disease_train, test_size=0.30, random_state=0, stratify=y_train
    )
    y_train = y_train == condition

    model = models.get_taxonomic_model(lopo=True, selector=True)
    n_bins = 2 if profile_name == "centrifuge" else 20
    model = ExplainableBoostingClassifier(n_jobs=-1, max_n_bins=n_bins)
    model.fit(X_train, y_train)

    X_test = features.loc[~query, columns]
    y_test = metadata.DISEASE[~query]

    new_metadata = pd.concat((metadata, metadata_adenoma), axis=0)
    new_metadata.drop_duplicates(inplace=True)

    probas_test = model.predict_proba(X_test)[:, 1]
    probas_test = pd.Series(probas_test, index=y_test.index, name="proba")
    data_test = pd.concat((probas_test, y_test), axis=1)

    probas_val = model.predict_proba(X_val)[:, 1]
    probas_val = pd.Series(probas_val, index=y_val.index, name="proba")
    data_val = pd.concat((probas_val, y_val), axis=1)

    data = pd.concat((data_val, data_test), axis=0)
    # data = pd.concat((data, new_metadata[PROJECT_COLUMN_NAME]), axis=1, join="inner")
    data["Project"] = new_metadata.loc[data.index, PROJECT_COLUMN_NAME]

    query = data.DISEASE.str.lower().str.contains("metastases")
    data.DISEASE[query] = "Metastases"

    query = data.DISEASE.str.contains("CRC;")
    data.DISEASE[query] = "CRC + Others"

    query = data.DISEASE.str.contains("adenoma;")
    data.DISEASE[query] = "Adenoma + Others"

    query = data.DISEASE.str.lower().str.contains("small")
    data.DISEASE[query] = "Small Adenoma"

    query = data.DISEASE.str.lower().str.contains(
        "adenoma"
    ) & ~data.DISEASE.str.lower().str.contains("small|others")
    data.DISEASE[query] = "Adenoma"

    query = data.DISEASE.str.contains("T2D")
    data.DISEASE[query] = "T2D + Others (non CRC)"

    adenoma_interpret_query = data.index[data[DISEASE_COLUMN_NAME] == "Adenoma"]
    X_adenoma = features.loc[adenoma_interpret_query, columns]

    ebm_local = model.explain_local(X_adenoma)

    explanations = [ebm_local.data(irow) for irow in range(X_adenoma.shape[0])]
    explanation_df = pd.concat(
        [pd.Series(x["scores"], index=x["names"], name="score") for x in explanations],
        axis=1,
    )
    explanation_df.columns = adenoma_interpret_query
    explanation_df = explanation_df.T

    adenoma_reg = [
        linregress(X_adenoma[col], explanation_df[col]) for col in X_adenoma.columns
    ]
    adenoma_coef = pd.Series(
        [reg[0] for reg in adenoma_reg], index=X_adenoma.columns, name="r_coeff"
    )
    adenoma_sig = pd.Series(
        [coef > 0 for coef in adenoma_coef], index=X_adenoma.columns, name="r_sign"
    )
    adenoma_pval = pd.Series(
        [reg[3] for reg in adenoma_reg], index=X_adenoma.columns, name="r_pvalue"
    )
    adenoma_pval_fdr = pd.Series(
        pystab.fdr(adenoma_pval), index=X_adenoma.columns, name="r_pvalue_fdr"
    )

    adenoma_rank = explanation_df.abs().sum().sort_values(ascending=False)
    adenoma_rank.name = "score"
    adenoma_analysis = pd.concat(
        (adenoma_rank, adenoma_coef, adenoma_sig, adenoma_pval, adenoma_pval_fdr),
        axis=1,
    )

    dataset_fpath = folder_path.joinpath(
        f"{condition}_{profile_name}_adenoma_explanations.tsv"
    )
    adenoma_analysis.to_csv(dataset_fpath, sep="\t", index_label="feature_id")

    ###################################################################

    query_ = metadata.DISEASE.isin([condition, "healthy"])
    disease_train = metadata.DISEASE[query_]
    y_ = metadata.DISEASE[query_] == condition
    X_ = features.loc[query_, :]
    metadata_adenoma = metadata.loc[~query_, :].copy()

    X_ = features.loc[query_, columns]

    small_l_healthy = []
    small_g_healthy = []
    healthy_l_adenoma = []
    small_l_adenoma = []

    n_split = 100

    for i in range(n_split):
        X_train, X_val, y_train, y_val = train_test_split(
            X_, disease_train, test_size=0.30, random_state=i, stratify=y_
        )
        y_train = y_train == condition

        # model = models.get_taxonomic_model(lopo=True, selector=True)
        model = ExplainableBoostingClassifier(
            n_estimators=32, n_jobs=-1, max_n_bins=n_bins
        )
        model.fit(X_train, y_train)

        X_test = features.loc[~query_, X_.columns]
        y_test = metadata.DISEASE[~query_]

        new_metadata = pd.concat((metadata, metadata_adenoma), axis=0)
        # new_metadata = new_metadata.loc[new_metadata.index.drop_duplicates(keep="first"), :]
        new_metadata.drop_duplicates(inplace=True)

        probas_test = model.predict_proba(X_test)[:, 1]
        probas_test = pd.Series(probas_test, index=y_test.index, name="proba")
        data_test = pd.concat((probas_test, y_test), axis=1)

        probas_val = model.predict_proba(X_val)[:, 1]
        probas_val = pd.Series(probas_val, index=y_val.index, name="proba")
        data_val = pd.concat((probas_val, y_val), axis=1)

        data = pd.concat((data_val, data_test), axis=0)
        # data = pd.concat((data, new_metadata[PROJECT_COLUMN_NAME]), axis=1, join="inner")
        data["Project"] = new_metadata.loc[data.index, PROJECT_COLUMN_NAME]

        query = data.DISEASE.str.lower().str.contains("metastases")
        data.DISEASE[query] = "Metastases"

        query = data.DISEASE.str.contains("CRC;")
        data.DISEASE[query] = "CRC + Others"

        query = data.DISEASE.str.contains("adenoma;")
        data.DISEASE[query] = "Adenoma + Others"

        query = data.DISEASE.str.lower().str.contains("small")
        data.DISEASE[query] = "Small Adenoma"

        query = data.DISEASE.str.lower().str.contains(
            "adenoma"
        ) & ~data.DISEASE.str.lower().str.contains("small|others")
        data.DISEASE[query] = "Adenoma"

        query = data.DISEASE.str.contains("T2D")
        data.DISEASE[query] = "T2D + Others (non CRC)"

        small = data.loc[data.DISEASE == "Small Adenoma", "proba"]
        healthy = data.loc[data.DISEASE == "healthy", "proba"]
        adenoma = data.loc[data.DISEASE == "Adenoma", "proba"]

        small_l_healthy.append(
            stats.mannwhitneyu(small, healthy, use_continuity=True, alternative="less")
        )
        small_g_healthy.append(
            stats.mannwhitneyu(
                small, healthy, use_continuity=True, alternative="greater"
            )
        )
        healthy_l_adenoma.append(
            stats.mannwhitneyu(
                healthy, adenoma, use_continuity=True, alternative="less"
            )
        )
        small_l_adenoma.append(
            stats.mannwhitneyu(small, adenoma, use_continuity=True, alternative="less")
        )

    adenoma_mwtest = {
        "small_l_healthy": small_l_healthy,
        "small_g_healthy": small_g_healthy,
        "healthy_l_adenoma": healthy_l_adenoma,
        "small_l_adenoma": small_l_adenoma,
    }

    mw_path = folder_path.joinpath(f"{condition}_{profile_name}_adenoma_mwtest.jbl")
    joblib.dump(adenoma_mwtest, mw_path)


if __name__ == "__main__":
    import sys

    _, this_condition, this_profile, this_path = sys.argv

    main(this_condition, this_profile, this_path)
