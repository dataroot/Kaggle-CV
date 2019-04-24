import pandas as pd

from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models.ldamulticore import LdaMulticore


def pipeline_lda(que: pd.DataFrame, dim: int) -> (Dictionary, TfidfModel, LdaMulticore):
    """
    Pipeline for training embeddings for questions via LDA algorithm
    on question titles and bodies

    :param que: raw questions.csv dataset
    :param dim: dimension of doc2vec embeddings to train
    :return: trained tags, industries embeddings and question's Doc2Vec model
    """
    lda_tokens = que['questions_whole'].apply(lambda x: x.split())

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
        passes=20, chunksize=1000, random_state=0
    )

    return lda_dic, lda_tfidf, lda_model
