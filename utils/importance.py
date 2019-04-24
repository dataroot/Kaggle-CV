import copy

import numpy as np
import pandas as pd

# Use for matplotlib PC for MAC OS
import matplotlib
matplotlib.use('TkAgg') 

from matplotlib import pyplot as plt

from sklearn.utils import shuffle
import keras


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
    for i, name in enumerate(fn['que'] + fn['pro']):
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


def plot_fi(fi, title='Feature importances via shuffle', xlabel='Change in loss after shuffling feature\'s values'):
    """
    Nicely plot Pandas DataFrame with feature importance
    """

    def get_color(feature: str):
        if feature.startswith('que'):
            if '_emb_' in feature:
                return 'b'
            else:
                return 'cornflowerblue'
        elif feature.startswith('pro'):
            if '_emb_' in feature:
                return 'firebrick'
            else:
                return 'indianred'
        else:
            return 'g'

    fi['color'] = fi.index.map(get_color)
    fig, ax = plt.subplots(figsize=(8, 20))
    plt.barh(fi.index, fi.importance, color=fi.color)
    plt.title(title)
    plt.xlabel(xlabel)
    ax.yaxis.tick_right()
    plt.show()
