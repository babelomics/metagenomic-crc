# coding: utf-8
"""
author: Carlos Loucera
email: carlos.loucera@juntadeandalucia.es

Analise pre-trained models.
"""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn import metrics
from sklearn.model_selection import LeaveOneGroupOut

from mlgut.stability import nogueria_test
from mlgut import utils
from mlgut.datasets import get_path
from mlgut.models import (
    compute_rbo_mat,
    compute_support_ebm,
    get_cp_support,
    get_lopo_support,
)

EXTENSIONS = ["pdf", "png", "svg"]

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


def load_crossproject(condition, profile, path):
    crossproject_fname = f"{condition}_{profile}_cross_project.jbl"
    crossproject_fpath = path.joinpath(crossproject_fname)
    crossproject_results = joblib.load(crossproject_fpath)
    crossproject_results = utils.rename_keys(crossproject_results, PROJECT_NAMES_DICT)

    return crossproject_results


def load_stability(condition, profile, path):
    stability_fname = f"{condition}_{profile}_stability.jbl"
    stability_fpath = path.joinpath(stability_fname)
    stability_results = joblib.load(stability_fpath)
    stability_results = utils.rename_keys(stability_results, PROJECT_NAMES_DICT)

    return stability_results


def compute_error(results, alpha=0.05, metric="roc_auc"):
    # Z matrix in Nogueira's stability paper
    metric_str = f"test_{metric}"
    scores = np.array(results[metric_str])
    n_splits = scores.size

    mean_score = np.mean(scores)
    ci = stats.norm.ppf(1 - alpha / 2) * (np.std(scores)) / np.sqrt(n_splits)

    return scores, mean_score, ci


def compute_stability(results, alpha=0.05):
    # Z matrix in Nogueira's stability paper

    support_matrix = [
        model["selector"].get_support() * 1 for model in results["estimator"]
    ]
    support_matrix = np.array(support_matrix)

    stab_res = nogueria_test(support_matrix, alpha=alpha)
    stability = stab_res.estimator
    stability_error = stab_res.error

    return support_matrix, stability, stability_error


def analyze_stability(
    stability_results, crossproject_results, condition, profile, path
):
    stability_results_df = {
        key: compute_stability(stability_results[key])[1:]
        for key in stability_results.keys()
    }
    stability_results_df = pd.DataFrame(
        stability_results_df, index=["estability", "error"]
    )
    stability_results_df = stability_results_df.T.sort_index()

    crossproject_results_df = {
        key: compute_stability(crossproject_results[key]["cv"])[1:]
        for key in crossproject_results.keys()
    }
    crossproject_results_df = pd.DataFrame(
        crossproject_results_df, index=["estability", "error"]
    )
    crossproject_results_df = crossproject_results_df.T.sort_index()

    roc_auc_stability = {
        key: compute_error(stability_results[key])[0]
        for key in stability_results.keys()
    }

    roc_auc_stability = pd.DataFrame(roc_auc_stability).melt(
        value_name="Mean AUROC", var_name="Project"
    )

    roc_auc_crossproject = {
        key: compute_error(crossproject_results[key]["cv"])[0]
        for key in crossproject_results.keys()
    }
    roc_auc_crossproject = pd.DataFrame(roc_auc_crossproject).melt(
        value_name="Mean AUROC", var_name="Project"
    )

    plot_stability(
        crossproject_results_df, stability_results_df, condition, profile, path
    )
    plot_error(roc_auc_crossproject, roc_auc_stability, condition, profile, path)

    return roc_auc_crossproject


def plot_stability(cp_df, stab_df, condition, profile, path):
    plt.style.use("fivethirtyeight")
    _, ax = plt.subplots(1, 1, figsize=(16, 9))
    plt.errorbar(
        x=cp_df.index,
        y=cp_df["estability"],
        yerr=cp_df["error"],
        label=["CV-stability"],
    )
    plt.errorbar(
        x=stab_df.index,
        y=stab_df["estability"],
        yerr=stab_df["error"],
        label=["RSSS-stability"],
    )
    ax.set_xlabel("Project")
    ax.set_ylabel("Stability")
    plt.legend(loc="lower right")
    plt.tight_layout()
    for ext in EXTENSIONS:
        fname = f"{condition}_{profile}_feature_selection_stability.{ext}"
        fpath = path.joinpath(fname)
        plt.savefig(fpath, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def plot_error(cp_df, stab_df, condition, profile, path):
    _, ax = plt.subplots(1, 1, figsize=(16, 9))

    sns.lineplot(
        x="Project",
        y="Mean AUROC",
        data=cp_df,
        ax=ax,
        err_style="bars",
        ci=95,
        label="CV-test",
    )
    sns.lineplot(
        x="Project",
        y="Mean AUROC",
        data=stab_df,
        ax=ax,
        err_style="bars",
        ci=95,
        label="RSSS-test",
    )
    plt.legend(loc="lower right")
    plt.tight_layout()
    for ext in EXTENSIONS:
        fname = f"{condition}_{profile}_feature_selection_stability_error.{ext}"
        fpath = path.joinpath(fname)
        plt.savefig(fpath, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def analyze_rank_stability(results, features, profile, condition, path):
    def compute_ebm_fi_by_project(results, key):
        ebm_fi_list = [
            pd.Series(compute_support_ebm(model)[1], index=features.columns)
            for model in results[key]["cv"]["estimator"]
        ]
        ebm_fi = pd.concat(ebm_fi_list, axis=1)

        return ebm_fi

    ebm_fi_by_project = {
        key: compute_ebm_fi_by_project(results, key) for key in results.keys()
    }

    dmat = {
        key: compute_rbo_mat(ebm_fi_by_project[key]) for key in ebm_fi_by_project.keys()
    }

    dmat = pd.DataFrame(dmat).melt(value_name="dRBO", var_name="Project")
    fname = f"{condition}_{profile}_rank_stability_mat.jbl"
    fpath = path.joinpath(fname)
    joblib.dump(dmat, fpath)

    plt.figure(figsize=(16, 9))
    sns.violinplot(x="Project", y="dRBO", data=dmat, order=PROJECT_ORDER)
    plt.ylim([-0.1, 1.1])
    plt.tight_layout()
    for ext in EXTENSIONS:
        fname = f"{condition}_{profile}_rank_stability.{ext}"
        fpath = path.joinpath(fname)
        plt.savefig(fpath, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def analyze_lopo_wo_oracle(results, features, metadata, profile, condition, control):
    query = metadata[DISEASE_COLUMN_NAME].isin([control, condition])
    projects = metadata.loc[query, PROJECT_COLUMN_NAME]

    lopo_mean = dict(zip(np.unique(projects), results["test_roc_auc"]))
    lopo_mean = utils.rename_keys(lopo_mean, PROJECT_NAMES_DICT)

    _, support_merged = get_lopo_support(results, features.columns)

    print(lopo_mean, np.mean(results["test_roc_auc"]))

    return lopo_mean, support_merged


def analyze_lopo_with_oracle(results, metadata, profile, condition, control):
    query = metadata[DISEASE_COLUMN_NAME].isin([control, condition])
    projects = metadata.loc[query, PROJECT_COLUMN_NAME]

    best = 0.0
    keep = {}
    best_support = []
    for i in results.keys():
        # top: at least in `i` LOPO trainings.
        lopo_mean_i = dict(zip(np.unique(projects), results[i]["cv"]["test_roc_auc"]))
        mean_i = np.mean(results[i]["cv"]["test_roc_auc"])
        lopo_mean_i = utils.rename_keys(lopo_mean_i, PROJECT_NAMES_DICT)
        if mean_i > best:
            best = mean_i
            keep = lopo_mean_i
            columns = results[i]["columns"]
            _, best_support = get_lopo_support(results[i]["cv"], columns)

        print(lopo_mean_i, mean_i)

    return keep, best_support


def plot_lopo(frame, support, profile, condition, path, oracle=True):
    if oracle:
        with_str = "with"
    else:
        with_str = "wo"
    print(frame)
    fname = f"{condition}_{profile}_lopo_{with_str} _oracle_support.tsv"
    fpath = path.joinpath(fname)
    support.to_csv(fpath, sep="\t")


def get_cross_project_data(names, profile, condition, results):

    l = []
    for project_id in results.keys():
        r = results[project_id]["outer"][
            [PROJECT_COLUMN_NAME, DISEASE_COLUMN_NAME, "decission"]
        ].copy()
        r[PROJECT_COLUMN_NAME] = r[PROJECT_COLUMN_NAME].replace(PROJECT_NAMES_DICT)
        r = r.groupby(PROJECT_COLUMN_NAME).apply(
            lambda x: metrics.roc_auc_score(
                x[DISEASE_COLUMN_NAME] == condition, x["decission"]
            )
        )
        r.name = project_id
        l.append(r)

    r = pd.concat(l, axis=1).T
    for project_id in results.keys():
        r.loc[project_id, project_id] = np.mean(
            results[project_id]["cv"]["test_roc_auc"]
        )

    fi, fi_merged = get_cp_support(results, names)

    return r, fi, fi_merged


def build_scoring_mat(cp_mat, lopo_wo_oracle, lopo_with_oracle):

    mat = cp_mat.copy()
    mat = mat.loc[PROJECT_ORDER, PROJECT_ORDER]
    mat.loc["Mean", :] = mat.mean(axis=0)
    lopo_wo_oracle_series = pd.Series(lopo_wo_oracle, name="LOPO")
    lopo_with_oracle_series = pd.Series(lopo_with_oracle, name="oLOPO")
    mat = mat.append(lopo_wo_oracle_series)
    mat = mat.append(lopo_with_oracle_series)
    mat["Mean"] = mat.mean(axis=1)

    return mat


def plot_scores(mat, condition, profile, path):

    plt.figure()
    ax = sns.heatmap(
        mat,
        annot=True,
        fmt=".2f",
        vmin=0.0,
        vmax=1.0,
        cbar=False,
        xticklabels=True,
        yticklabels=True,
    )
    ax.xaxis.set_ticks_position("top")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    for ext in EXTENSIONS:
        fname = f"{condition}_{profile}_score_matrix.{ext}"
        fpath = path.joinpath(fname)
        plt.savefig(fpath, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def load_lopo(condition, profile, path, oracle=False):
    if oracle:
        with_str = "with"
    else:
        with_str = "wo"

    fname = f"{condition}_{profile}_lopo_{with_str}_oracle.jbl"
    fpath = path.joinpath(fname)
    results = joblib.load(fpath)

    return results


def build_analysis(features, metadata, profile, condition, control, path):
    results_stab = load_stability(condition, profile, path)
    results_cp = load_crossproject(condition, profile, path)
    results_lopo_wo_oracle = load_lopo(condition, profile, path, oracle=False)
    results_lopo_with_oracle = load_lopo(condition, profile, path, oracle=True)

    metadata_ = metadata.copy()
    metadata_[PROJECT_COLUMN_NAME] = metadata_[PROJECT_COLUMN_NAME].replace(
        PROJECT_NAMES_DICT
    )

    cp_mat, cp_fi, cp_fi_merged = get_cross_project_data(
        features.columns, profile, condition, results_cp
    )
    cp_fi.to_csv(path.joinpath(f"{condition}_{profile}_cp_support.tsv"), sep="\t")
    cp_fi_merged.to_csv(
        path.joinpath(f"{condition}_{profile}_cp_support_merge.tsv"), sep="\t"
    )

    lopo_wo_oracle, support_lopo_wo_oracle = analyze_lopo_wo_oracle(
        results_lopo_wo_oracle, features, metadata, profile, condition, control
    )
    lopo_with_oracle, support_lopo_with_oracle = analyze_lopo_with_oracle(
        results_lopo_with_oracle, metadata, profile, condition, control
    )

    score_mat = build_scoring_mat(cp_mat, lopo_wo_oracle, lopo_with_oracle)
    plot_scores(score_mat, condition, profile, path)

    plot_lopo(
        lopo_wo_oracle, support_lopo_wo_oracle, profile, condition, path, oracle=False
    )
    plot_lopo(
        lopo_with_oracle,
        support_lopo_with_oracle,
        profile,
        condition,
        path,
        oracle=True,
    )

    analyze_stability(results_stab, results_cp, condition, profile, path)
    analyze_rank_stability(results_cp, features, profile, condition, path)
