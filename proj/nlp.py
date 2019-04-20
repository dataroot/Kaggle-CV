import random

import pandas as pd

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models.ldamulticore import LdaMulticore


def train_d2v(df: pd.DataFrame, target: str, features: list, dim: int) -> (Doc2Vec, dict):
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
    d2v = Doc2Vec(prepared, vector_size=dim, workers=4, epochs=10, dm=0)
    docvecs = {d2v.docvecs.index2entity[i]: d2v.docvecs.vectors_docs[i]
               for i in range(len(d2v.docvecs.index2entity))}
    return d2v, docvecs


def train_lda(df: pd.DataFrame, target: str, dim: int) -> (Dictionary, TfidfModel, LdaMulticore):
    """
    Train LdaMulticore model on provided data

    :param df: data to work with
    :param target: feature name to be used
    :param dim: dimension of embedding vectors to train
    :return: trained LdaMulticore model
    """
    lda_tokens = df[target].apply(lambda x: x.split())

    # create Dictionary and train it on text corpus
    lda_dic = Dictionary(lda_tokens)
    lda_dic.filter_extremes(no_below=10, no_above=0.6, keep_n=8000)
    lda_corpus = [lda_dic.doc2bow(doc) for doc in lda_tokens]

    # create TfidfModel and train it on text corpus 
    lda_tfidf = TfidfModel(lda_corpus)
    lda_corpus = lda_tfidf[lda_corpus]

    # create LDA Model and train it on text corpus
    lda_model = LdaMulticore(
        lda_corpus, num_topics=dim, id2word=lda_dic, workers=4,
        passes=20, chunksize=1000, alpha=0.02, random_state=0
    )

    return lda_dic, lda_tfidf, lda_model


'''
def vis_d2v(d2v: Doc2Vec):
    """
    Visualize with T-SNE embeddings trained with doc2vec
    """
    proj = TSNE(n_components=2, verbose=1).fit_transform(d2v.docvecs.vectors_docs)
    _, ax = plt.subplots(figsize=(60, 60))
    plt.scatter(proj[:, 0], proj[:, 1], alpha=0.5)
    for i, name in enumerate(d2v.docvecs.index2entity):
        ax.annotate(name, (proj[i, 0], proj[i, 1]))
    plt.show()
'''


# TODO: update to consider professional's tags

def pipeline_d2v(que: pd.DataFrame, ans: pd.DataFrame, pro: pd.DataFrame, tags: pd.DataFrame,
                 dim: int) -> (dict, dict, Doc2Vec):
    """
    Pipeline for training embeddings for
    professional's industries and question's tags via doc2vec algorithm
    on question titles, bodies, answer bodies, names of tags, professional industries and headlines

    :param que: raw questions.csv dataset
    :param ans: raw answers.csv dataset
    :param pro: raw professionals.csv dataset
    :param tags: tags.csv merged with tag_questions.csv
    :param dim: dimension of doc2vec embeddings to train
    :return: trained tags, industries embeddings and question's Doc2Vec model
    """

    # merge questions, tags, answers and professionals
    que_tags = que.merge(tags, left_on='questions_id', right_on='tag_questions_question_id')
    ans_que_tags = ans.merge(que_tags, left_on="answers_question_id", right_on="questions_id")
    df = ans_que_tags.merge(pro, left_on='answers_author_id', right_on='professionals_id')

    text_features = ['questions_title', 'questions_body', 'answers_body',
                     'tags_tag_name', 'professionals_industry', 'professionals_headline']

    # train and save question's tags embeddings
    _, tags_embs = train_d2v(df, 'tags_tag_name', text_features, dim)

    # aggregate all the tags in one string for same questions
    que_tags = que_tags[['questions_id', 'tags_tag_name']].groupby(by='questions_id', as_index=False) \
        .aggregate(lambda x: ' '.join(x))

    # merge questions, aggregated tags, answers and professionals
    que_tags = que.merge(que_tags, on='questions_id')
    ans_que_tags = ans.merge(que_tags, left_on="answers_question_id", right_on="questions_id")
    df = ans_que_tags.merge(pro, left_on='answers_author_id', right_on='professionals_id')

    # train and save professional's industries embeddings
    _, inds_embs = train_d2v(df, 'professionals_industry', text_features, dim)

    que_tags['questions_whole'] = que_tags['questions_title'] + ' ' + que_tags['questions_body']

    ques_d2v, _ = train_d2v(que_tags, 'questions_id', ['questions_whole'], 10)

    return tags_embs, inds_embs, ques_d2v


def pipeline_lda(que: pd.DataFrame, tags: pd.DataFrame, dim: int) -> (Dictionary, TfidfModel, LdaMulticore):
    """
    Pipeline for training embeddings for questions via LDA algorithm
    on question titles and bodies

    :param que: raw questions.csv dataset
    :param tags: tags.csv merged with tag_questions.csv
    :param dim: dimension of doc2vec embeddings to train
    :return: trained tags, industries embeddings and question's Doc2Vec model
    """
    # aggregate all the tags in one string for same questions
    que_tags = tags[['tag_questions_question_id', 'tags_tag_name']] \
        .groupby(by='tag_questions_question_id', as_index=False).aggregate(lambda x: ' '.join(x))

    # merge questions and aggregated tags
    que_tags = que.merge(que_tags, left_on='questions_id', right_on='tag_questions_question_id')
    que_tags['questions_whole'] = que_tags['questions_title'] + ' ' + que_tags['questions_body']

    lda_dic, lda_tfidf, lda_model = train_lda(que_tags, 'questions_whole', dim)

    return lda_dic, lda_tfidf, lda_model
