"""General utilities for the experiments."""

import logging
from collections import ChainMap
from typing import Union, List, Tuple, Any
import re
import os
import numpy as np
import scipy as sp
import pandas as pd
import scipy.stats
import scipy.linalg
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm

from emutils.utils import keydefaultdict, attrdict
from emutils.file import load_pickle, save_pickle


def safe_mean(a):
    return np.mean(a[~np.isnan(a)])


def replace_nan(a, replace_with=1.0):
    return np.where(~np.isnan(a), a, replace_with)


def safe_std(a):
    return np.std(a[~np.isnan(a)])


def failure_nan(a):
    return (np.isnan(a).sum() / len(a))


DATASET_NAMES = keydefaultdict(
    lambda k: k, {
        'heloc': 'HELOC (Home Equity Line of Credit)',
        'lendingclub': 'Lending Club (2007-2018)',
        'wines': 'Wine Quality (White)',
    })

MARKERS_SHAPES = ['s', 'o', '^', '*', 'P', 'D', '<', '>', 'v', '.', 'h', 'H', 'X', '8', '1', '2', '3', '4', '+', 'x']


def get_markers(n):
    markers_ = []
    while True:
        if len(markers_) == n:
            break
        markers_ += MARKERS_SHAPES[:(n - len(markers_))]
    return markers_


def get_colors(n, cmap=None):
    if cmap is not None or n > 20:
        return [tuple(c) for c in mpl.cm.get_cmap(cmap)(np.linspace(0, 1 - 1 / n, n))[:, :-1]]
    elif n <= 10:
        return list(mpl.cm.get_cmap('tab10').colors[:n])
    elif n <= 20:
        return list(mpl.cm.get_cmap('tab20').colors[:n])


def dataset_to_name(dataset):
    return DATASET_NAMES[dataset]


def model_version_to_name(model_version):
    return "Monotonic" if 'mono' in model_version else "Non-Monotonic"


def result_version_to_name(result_version):
    if '_test' in result_version:
        return 'Test Sample'
    elif 'iperclose' in result_version:
        return 'Iper Close (<2.5%)'
    if 'superclose' in result_version:
        return 'Mega Close (<10%)'
    if 'veryclose' in result_version:
        return 'Very Close (<20%)'
    elif 'close' in result_version:
        return 'Close (<50%)'
    elif 'veryfar' in result_version:
        return 'Very Far (>80%)'
    elif 'far' in result_version:
        return 'Far (>50%)'
    else:
        return 'All'


METHOD_NAMES = {
    ('SHAP', 'base_med'): 'SHAP Median',
    ('SHAP', 'base_mean'): 'SHAP Mean',
    ('SHAP', 'base_medgood'): 'SHAP Median Good',
    ('SHAP', 'base_meangood'): 'SHAP Mean Good',
    ('SHAP', 'training'): 'SHAP TRAIN',
    ('SHAP', 'diff_pred'): 'SHAP D-PRED',
    ('SHAP', 'cone'): 'SHAP D-PRED $^*$',
    ('SHAP', 'diff_label'): 'SHAP D-LAB',
    ('SHAP', 'base_mean'): 'SHAP D-MEAN',
    ('SHAP', 'training_100'): 'SHAP TRAIN (n = 100)',
    ('SHAP', 'diff_pred_100'): 'SHAP D-PRED (n = 100)',
    ('SHAP', 'diff_label_100'): 'SHAP D-LAB (n = 100)',
}


def method_to_name(method, trend=True, norm=True, details=True):
    if isinstance(method, str):
        method = ('SHAP', method, None)

    if isinstance(method, tuple) and len(method) == 2:
        method = tuple(list(method) + [None])

    if len(method) != 3:
        return str(method)

    m_type, m_name, m_trend = method
    method2 = method[:2]

    def _method_to_name():
        mname = method2
        if mname in METHOD_NAMES:
            return METHOD_NAMES[method2]

        if m_type == 'SHAP':
            if "_FREQ" in m_name:
                return "FREQ"
            elif "_DIST" in m_name:
                return "DIST"
            elif 'knn' in m_name:
                return f"CF-SHAP ${int(re.findall('knn([0-9]+)_.*', m_name)[0])}$-NN"
            else:
                return f'Unknown ({m_name})'

    def _method_to_trend():
        if m_trend is None or trend is False:
            return ""
        elif m_trend == 'random':
            return " R"
        elif m_trend == 'local':
            return " L"
        elif m_trend == 'global':
            return " G"
        else:
            return " UnknTrend"

    def method_norm():
        if norm:
            m_norm = re.findall('.*_[a-z](L[0-9])', m_name)
            m_norm = m_norm[0] if m_norm else ""
            m_trans = re.findall('.*_([a-z])L[0-9]', m_name)
            m_trans = m_trans[0].upper() if m_trans else ""

            if m_norm and m_trans:
                return m_trans + "+" + m_norm
            else:
                return m_trans + m_norm
        else:
            return ""

    def method_cone():
        if 'cone' in m_name:
            return "$^*$"
        else:
            return ""

    def method_diverse():
        if 'diverse' in m_name:
            return r"$^{\dagger}$"
        else:
            return ""

    name_ = _method_to_name()
    name_ += method_norm()
    name_ += method_cone()
    name_ += method_diverse()
    name_ += _method_to_trend()

    if not details:
        # Remove parenthesis
        name_ = re.sub(r"(\ )?\(.*?\)", "", name_)

    return name_


def result_filename(args,
                    result_name,
                    dataset=None,
                    data_version=None,
                    model_version=None,
                    model_type=None,
                    results_version=None,
                    ext='pkl'):
    if results_version is None:
        results_version = args.results_version

    mrn = model_run_name(args,
                         dataset=dataset,
                         data_version=data_version,
                         model_version=model_version,
                         model_type=model_type)

    return f"{args.results_path}/{mrn}_{result_name}{results_version}.{ext}"


def model_run_name(args, dataset=None, data_version=None, model_version=None, model_type=None):
    if dataset is None:
        dataset = args.dataset
    if data_version is None:
        data_version = args.data_version
    if model_version is None:
        model_version = args.model_version
    if model_type is None:
        model_type = args.model_type
    return f"{dataset}_D{data_version}M{model_version}_{model_type}"


def load_explanations(args,
                      dataset=None,
                      data_version=None,
                      model_version=None,
                      model_type=None,
                      results_version=None,
                      backgrounds=False):
    e = attrdict(
        metadata=load_pickle(result_filename(args,
                                             result_name='meta_all',
                                             dataset=dataset,
                                             data_version=data_version,
                                             model_version=model_version,
                                             model_type=model_type,
                                             results_version=results_version),
                             verbose=0),
        values=load_pickle(result_filename(args,
                                           result_name='values_all',
                                           dataset=dataset,
                                           data_version=data_version,
                                           model_version=model_version,
                                           model_type=model_type,
                                           results_version=results_version),
                           verbose=0),
        trends=load_pickle(result_filename(args,
                                           result_name='trends_all',
                                           dataset=dataset,
                                           data_version=data_version,
                                           model_version=model_version,
                                           model_type=model_type,
                                           results_version=results_version),
                           verbose=0),
    )

    if backgrounds:
        e.backgrounds = load_pickle(result_filename(args,
                                                    result_name='backgrounds_all',
                                                    dataset=dataset,
                                                    data_version=data_version,
                                                    model_version=model_version,
                                                    model_type=model_type,
                                                    results_version=results_version),
                                    verbose=0),
    return e


def profiling_filename(
    args,
    dataset=None,
    data_version=None,
    model_version=None,
    model_type=None,
    ext='pkl',
):
    mrn = model_run_name(args,
                         dataset=dataset,
                         data_version=data_version,
                         model_version=model_version,
                         model_type=model_type)
    return f"{args.results_path}/{mrn}_profiling.{ext}"


def load_data(args, dataset=None, data_version=None):
    if dataset is None:
        dataset = args.dataset
    if data_version is None:
        data_version = args.data_version

    DATA_RUN_NAME = f"{dataset}_D{data_version}"

    TRAIN_X_FILENAME = f"{args.data_path}/{DATA_RUN_NAME}_Xtrain.pkl"
    TEST_X_FILENAME = f"{args.data_path}/{DATA_RUN_NAME}_Xtest.pkl"
    TRAIN_Y_FILENAME = f"{args.data_path}/{DATA_RUN_NAME}_ytrain.pkl"
    TEST_Y_FILENAME = f"{args.data_path}/{DATA_RUN_NAME}_ytest.pkl"

    X_train = load_pickle(TRAIN_X_FILENAME)
    X_test = load_pickle(TEST_X_FILENAME)
    y_train = load_pickle(TRAIN_Y_FILENAME)
    y_test = load_pickle(TEST_Y_FILENAME)
    X = pd.concat([X_train, X_test], axis=0)
    y = pd.concat([y_train, y_test], axis=0)

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    return X, y, X_train, X_test, y_train, y_test


def load_data_and_model(dataset, data_version, model_version, args):
    DATA_RUN_NAME = f"{dataset}_D{data_version}"
    MODEL_RUN_NAME = f"{DATA_RUN_NAME}M{model_version}_xgb"

    TRAIN_X_FILENAME = f"{args.data_path}/{DATA_RUN_NAME}_Xtrain.pkl"
    TEST_X_FILENAME = f"{args.data_path}/{DATA_RUN_NAME}_Xtest.pkl"
    TRAIN_Y_FILENAME = f"{args.data_path}/{DATA_RUN_NAME}_ytrain.pkl"
    TEST_Y_FILENAME = f"{args.data_path}/{DATA_RUN_NAME}_ytest.pkl"
    MODELWRAPPER_FILENAME = f"{args.model_path}/{MODEL_RUN_NAME}_model.pkl"

    X_train = load_pickle(TRAIN_X_FILENAME, verbose=0)
    X_test = load_pickle(TEST_X_FILENAME, verbose=0)
    y_train = load_pickle(TRAIN_Y_FILENAME, verbose=0)
    y_test = load_pickle(TEST_Y_FILENAME, verbose=0)
    X = pd.concat([X_train, X_test], axis=0)
    y = pd.concat([y_train, y_test], axis=0)

    model = load_pickle(MODELWRAPPER_FILENAME, verbose=0)

    return attrdict(
        model=model,
        X=X,
        y=y,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )


def plt_arrow(*args, **kwargs):
    if 'label' in kwargs:
        plt.scatter(
            [],
            [],
            marker=r'$\rightarrow$',
            label=kwargs['label'],
            color=kwargs['color'] if 'color' in kwargs else (kwargs['facecolor'] if 'facecolor' in kwargs else 'black'),
            s=100,
        )  # dummy scatter to add an item to the legend
        del kwargs['label']
    return plt.arrow(*args, **kwargs)
