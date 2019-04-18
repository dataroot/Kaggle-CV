import random
import pickle

import pandas as pd

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from utils import TextProcessor


def train(df: pd.DataFrame, target: str, features: list, dim: int) -> Doc2Vec:
    """
    Train Doc2Vec object on provided data

    :param df: data to work with
    :param target: column name of target entity in df to train embeddings for
    :param features: list of feature names to be used for training
    :param dim: dimension of embedding vectors to train
    :return: trained Doc2Vec object
    """
    prepared = []
    for feature in features:
        if feature != target:
            prepared += [TaggedDocument(row[feature].split(), [row[target]])
                         for i, row in df[[feature, target]].drop_duplicates().iterrows()]
        else:
            prepared += [TaggedDocument(s.split(), [s]) for s in df[target].drop_duplicates()]
    # shuffle prepared data, just in case
    prepared = random.sample(prepared, len(prepared))
    return Doc2Vec(prepared, vector_size=dim, workers=4, epochs=10, dm=0)


def save(d2v: Doc2Vec, prefix: str):
    """
    Serialize dict with mapping from entity to it's doc2vec embedding
    and save Doc2Vec object itself
    """
    d2v.save(prefix + '.d2v')
    docvecs = {d2v.docvecs.index2entity[i]: d2v.docvecs.vectors_docs[i]
               for i in range(len(d2v.docvecs.index2entity))}
    with open(prefix + '_embs.pkl', 'wb') as file:
        pickle.dump(docvecs, file)


def vis(d2v: Doc2Vec):
    """
    Visualize with T-SNE embeddings trained with doc2vec
    """
    proj = TSNE(n_components=2, verbose=1).fit_transform(d2v.docvecs.vectors_docs)
    fig, ax = plt.subplots(figsize=(60, 60))
    plt.scatter(proj[:, 0], proj[:, 1], alpha=0.5)
    for i, name in enumerate(d2v.docvecs.index2entity):
        ax.annotate(name, (proj[i, 0], proj[i, 1]))
    plt.show()


# TODO: update to consider professional's tags

def pipeline(que: pd.DataFrame, ans: pd.DataFrame, pro: pd.DataFrame, tags: pd.DataFrame, dim: int, path: str):
    """
    Pipeline for training and saving to drive embeddings for
    professional's industries and question's tags via doc2vec algorithm
    on question titles, bodies, answer bodies, names of tags, professional industries and headlines

    :param que: raw questions.csv dataset
    :param ans: raw answers.csv dataset
    :param pro: raw professionals.csv dataset
    :param tags: tags.csv merged with tag_questions.csv
    :param dim: dimension of doc2vec embeddings to train
    """
    features = ['questions_title', 'questions_body', 'answers_body',
                'tags_tag_name', 'professionals_industry', 'professionals_headline']

    # pre-process all textual columns in data
    tp = TextProcessor(path)
    for df, column in zip([que, que, ans, tags, pro, pro], features):
        df[column] = df[column].apply(tp.process, allow_stopwords=(column == 'tags_tag_name'))

    # merge questions, tags, answers and professionals
    que_tags = que.merge(tags, left_on='questions_id', right_on='tag_questions_question_id')
    ans_que_tags = ans.merge(que_tags, left_on="answers_question_id", right_on="questions_id")
    df = ans_que_tags.merge(pro, left_on='answers_author_id', right_on='professionals_id')

    # train and save question's tags embeddings
    # d2v = train(df, 'tags_tag_name', features, dim)
    # save(d2v, path + 'tags')

    # aggregate all the tags in one string for same questions
    que_tags = que_tags[['questions_id', 'tags_tag_name']].groupby(by='questions_id', as_index=False) \
        .aggregate(lambda x: ' '.join(x))

    # merge questions, aggregated tags, answers and professionals
    que_tags = que.merge(que_tags, on='questions_id')
    ans_que_tags = ans.merge(que_tags, left_on="answers_question_id", right_on="questions_id")
    df = ans_que_tags.merge(pro, left_on='answers_author_id', right_on='professionals_id')

    # train and save professional's industries embeddings
    # d2v = train(df, 'professionals_industry', features, dim)
    # save(d2v, path + 'industries')

    que_tags['questions_all'] = que_tags['questions_title'] + ' ' + que_tags['questions_body']
    # d2v = train(que_tags, 'questions_id', ['questions_all'], 10)
    # save(d2v, path + 'questions')
