import copy

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from jupyterthemes import jtplot
from tqdm import tqdm

from sklearn.utils import shuffle
import keras

jtplot.style('gruvboxd')


def permutation_importance(model: keras.models.Model, x_que: np.ndarray, x_pro: np.ndarray, y: np.ndarray,
                           fn: dict, n_trials: int) -> pd.DataFrame:
    """
    Calculate model feature importance via random permutations of feature values

    :param model: model to evaluate
    :param x_que: pre-processed questions data
    :param x_pro: pre-processed professionals data
    :param y: target labels
    :param fn: dict with feature names of both questions and professionals
    :param n_trials: number of shuffles for each feature
    :return: Pandas DataFrame with importance of each feature
    """
    # model performance on normal, non-shuffled data
    base_loss, base_acc = model.evaluate([x_que, x_pro], y)
    losses = []
    for i, name in enumerate(tqdm(fn['que'] + fn['pro'], desc="Feature importance")):
        loss = 0
        for j in range(n_trials):
            x_que_i, x_pro_i = copy.deepcopy(x_que), copy.deepcopy(x_pro)

            if name in fn['que']:
                x_que_i[:, i] = shuffle(x_que_i[:, i])
            else:
                x_pro_i[:, i - len(fn['que'])] = shuffle(x_pro_i[:, i - len(fn['que'])])
            loss += model.evaluate([x_que_i, x_pro_i], y, verbose=0)[0]

        losses.append(loss / n_trials)

    fi = pd.DataFrame({'importance': losses}, index=fn['que'] + fn['pro'])
    fi.sort_values(by='importance', inplace=True, ascending=True)
    fi['importance'] -= base_loss

    return fi


def plot_fi(fi, fn, title='Feature importances via shuffle', xlabel='Change in loss after shuffling feature\'s values'):
    """
    Nicely plot Pandas DataFrame with feature importance
    """
    fi['color'] = 'b'
    fi.loc[fi.index.isin(fn['text']), 'color'] = 'r'
    fig, ax = plt.subplots(figsize=(8, 20))
    plt.barh(fi.index, fi.importance, color=fi.color)
    plt.title(title)
    plt.xlabel(xlabel)
    ax.yaxis.tick_right()
    plt.show()


def vis_emb(model, layer, names, figsize, colors, title, s=None):
    """
    Visualize embeddings of a single feature
    """
    emb = (model.get_layer(layer).get_weights()[0])
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(emb[:, 0], emb[:, 1], c=colors, s=s)
    for i, name in enumerate(names):
        ax.annotate(name, (emb[i, 0], emb[i, 1]))
    plt.title(title)
