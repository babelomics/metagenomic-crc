# coding: utf-8
"""
author: Carlos Loucera
email: carlos.loucera@juntadeandalucia.es

Run LOPO significance anlysis.
"""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn import metrics
from sklearn.model_selection import LeaveOneGroupOut

import mlgut.stability as stab
from mlgut import utils
from mlgut.datasets import get_path
from mlgut.models import (
    compute_rbo_mat,
    compute_support_ebm,
    get_cp_support,
    get_lopo_support,
)
import sys


def run_train(profiles):
    pass


def run_plot(profiles):
    pass


def run(profiles, mode):
    if mode == "train":
        run_train(profiles)
    elif mode == "plot":
        run_plot(profiles)
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    _, profiles, mode = sys.argv

    run(profiles, mode)
