import copy

import pandas as pd
from matplotlib import pyplot as plt

from sklearn.utils import shuffle
from tqdm import tqdm_notebook as tqdmn


def permutation_importance(is_content_related: bool, model, batch, y, fn, que_sep: int, pro_sep: int, n_trials: int = 3):
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
    base_loss, base_acc = model.evaluate(batch, y)
    
    que_inputs, pro_inputs = batch
    
    if is_content_related:
        que_offset, pro_offset = 0, 0
    else:
        que_offset, pro_offset = que_sep, pro_sep
    
    losses = []
    tuples = [(fn['que'], que_inputs, que_offset, 0), (fn['pro'], pro_inputs, pro_offset, 1)]
    
    for names, arr, offset, order_in_batch in tuples:
        for i in range(len(names)):
            loss = 0
            for _ in range(n_trials):
                arr_copy = copy.deepcopy(arr)
                arr_copy[:, offset + i] = shuffle(arr_copy[:, offset + i])
                
                batch_copy = copy.deepcopy(batch)
                batch_copy[order_in_batch] = arr_copy
                
                loss += model.evaluate(batch_copy, y, verbose=0)[0]
            losses.append(loss / n_trials)
    
    feature_names = fn['que'] + fn['pro']
    fi = pd.DataFrame({'importance': losses}, index=feature_names)
    fi.sort_values(by='importance', inplace=True, ascending=True)
    fi['importance'] -= base_loss
    
    return fi


def plot_fi(fi, fn, title='Feature importances via shuffle', xlabel='Change in loss after shuffling feature values'):
    """
    Nicely plot Pandas DataFrame with feature importances
    """
    fi.loc[fi.index.isin(fn['que']), 'color'] = 'b'
    fi.loc[fi.index.isin(fn['pro']), 'color'] = 'r'
    fig, ax = plt.subplots(figsize=(8, 20))
    plt.barh(fi.index, fi.importance, color=fi.color)
    plt.title(title)
    plt.xlabel(xlabel)
    ax.yaxis.tick_right()


def vis_emb(model, layer, names, figsize, colors, title, s=None):
    """
    Visualize embeddings of a single feature
    """
    emb = model.get_layer(layer).get_weights()[0]
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(emb[:, 0], emb[:, 1], c=colors, s=s)
    for i, name in enumerate(names):
        ax.annotate(name, (emb[i, 0], emb[i, 1]))
    plt.title(title)

