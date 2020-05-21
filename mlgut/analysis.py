# coding: utf-8
"""
author: Carlos Loucera
email: carlos.loucera@juntadeandalucia.es

Analise pre-trained models.
"""
import csv
import joblib
import numpy as np
from scipy import stats
import mlgut.stability as stab
from mlgut.datasets import get_path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlgut.models import compute_rbo_mat, compute_support_ebm


EXTENSIONS = ["pdf", "png", "svg"]

PROJECT_NAMES_DICT = {
    "PRJNA389927": "Hannigan",
    "PRJEB12449": "Vogtmann",
    "PRJEB6070": "Zeller",
    "PRJEB7774": "Feng",
    "PRJEB10878": "Yu",
    "PRJEB12449": "Vogtmann",
    "PRJNA447983": "Thomas0",
    "PRJEB27928": "Thomas1",
}

PROJECT_ORDER = sorted(PROJECT_NAMES_DICT.values())
DISEASE_COLUMN_NAME = "DISEASE"
PROJECT_COLUMN_NAME = "SECONDARY_STUDY_ID"
RESULTS_PATH = get_path("results")




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

    stab_res = stab.confidenceIntervals(support_matrix, alpha=alpha)
    stability = stab_res["stability"]
    stability_error = stab_res["stability"] - stab_res["lower"]

    return support_matrix, stability, stability_error



def analyze_stability(features, metadata, profile, condition, path):
    stability_fname = f"{condition}_{profile}_stability.jbl"
    stability_fpath = path.joinpath(stability_fname)
    stability_results = joblib.load(stability_fpath)
    for key in stability_results.keys():
        stability_results[PROJECT_NAMES_DICT[key]] = stability_results.pop(key)

    stability_results_df = {
        key: compute_stability(stability_results[key])[1:]
        for key in stability_results.keys()
    }
    stability_results_df = pd.DataFrame(
        stability_results_df, index=["estability", "error"]
    )
    stability_results_df = stability_results_df.T.sort_index()

    crossproject_fname = f"{condition}_{profile}_cross_project.jbl"
    crossproject_fpath = path.joinpath(crossproject_fname)
    crossproject_results = joblib.load(crossproject_fpath)
    for key in crossproject_results.keys():
        crossproject_results[PROJECT_NAMES_DICT[key]] = crossproject_results.pop(key)

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

    plot_stability(crossproject_results_df, stability_results_df, path)
    plot_error(roc_auc_crossproject, roc_auc_stability, path)


def plot_stability(cp_df, stab_df, path):
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
        fname = f"feature_selection_stability.{ext}"
        fpath = path.joinpath(fname)
        plt.savefig(fpath, dpi=300, bbox_inches="tight", pad_inches=0)


def plot_error(cp_df, stab_df, path):
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
        fname = f"feature_selection_stability_error.{ext}"
        fpath = path.joinpath(fname)
        plt.savefig(fpath, dpi=300, bbox_inches="tight", pad_inches=0)


def analyze_rank_stability(features, metadata, profile, condition, path):
    pass


def build_analysis(features, metadata, profile, condition, path):
    metadata_ = metadata.copy()
    metadata_[PROJECT_COLUMN_NAME] = metadata_[PROJECT_COLUMN_NAME].replace(
        PROJECT_NAMES_DICT
    )
    analyze_stability(features, metadata_, profile, condition, path)
    analyze_rank_stability(features, metadata_, profile, condition, path)
