import copy

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from jupyterthemes import jtplot
from tqdm import tqdm

from sklearn.utils import shuffle
import keras

jtplot.style('gruvboxd')


def permutation_importance(model: keras.models.Model, batch: np.ndarray, y: np.ndarray, qfn: np.array, pfn: np.array,
                           que_mask: np.ndarray, pro_mask: np.ndarray, n_trials: int = 3) -> pd.DataFrame:
    """
    Calculate model feature importance via random permutations of feature values

    :param model: model to evaluate
    :param batch: pre-processed batch of data
    :param y: target labels
    :param qfn: numpy array with question feature names
    :param pfn: numpy array with professional feature names
    :param que_mask: mask of question features
    :param pro_mask: mask of professional features
    :param n_trials: number of shuffles for each feature
    :return: Pandas DataFrame with importance of each feature
    """
    print("Begin2")
    # model performance on normal, non-shuffled data
    base_loss, base_acc = model.evaluate(batch, y)
    
    losses = []
    qfa, pfa = batch
    tuples = [(qfa, que_mask, 0), (pfa, pro_mask, 1)]
    
    for fa, mask, batch_order in tuples:
        for i, in_mask in enumerate(mask):
            if in_mask:
                loss = 0
                for _ in range(n_trials):
                    fa_copy = copy.deepcopy(fa)
                    fa_copy[:, i] = shuffle(fa_copy[:, i])
                    
                    batch_copy = copy.deepcopy(batch)
                    batch_copy[batch_order] = fa_copy
                    
                    loss += model.evaluate(batch_copy, y, verbose=0)[0]
                losses.append(loss / n_trials)
    
    fn = np.concatenate([qfn[que_mask], pfn[pro_mask]])
    fi = pd.DataFrame({'importance': losses}, index=fn)
    fi.sort_values(by='importance', inplace=True, ascending=True)
    fi['importance'] -= base_loss
    
    return fi


def plot_fi(fi, qfn, pfn, title='Feature importance via shuffle', xlabel='Change in loss after shuffling feature values'):
    """
    Nicely plot Pandas DataFrame with feature importance
    """
    fi.loc[fi.index.isin(qfn), 'color'] = 'b'
    fi.loc[fi.index.isin(pfn), 'color'] = 'r'
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
