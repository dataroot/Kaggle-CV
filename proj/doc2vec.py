import random
import pickle

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from utils import TextProcessor


def train(df, target, features, dim):
    prepared = []
    for feature in features:
        if feature != target:
            prepared += [TaggedDocument(row[feature].split(), [row[target]])
                         for i, row in df[[feature, target]].drop_duplicates().iterrows()]
        else:
            prepared += [TaggedDocument(s.split(), [s]) for s in df[target].drop_duplicates()]
    prepared = random.sample(prepared, len(prepared))
    return Doc2Vec(prepared, vector_size=dim, workers=4, iter=20, dm=0)


def save(d2v, prefix):
    d2v.save(prefix + '.d2v')
    docvecs = {d2v.docvecs.index2entity[i]: d2v.docvecs.vectors_docs[i]
               for i in range(len(d2v.docvecs.index2entity))}
    with open(prefix + '_embs.pkl', 'wb') as file:
        pickle.dump(docvecs, file)


def vis(d2v):
    proj = TSNE(n_components=2, verbose=1).fit_transform(d2v.docvecs.vectors_docs)
    fig, ax = plt.subplots(figsize=(60, 60))
    plt.scatter(proj[:, 0], proj[:, 1], alpha=0.5)
    for i, name in enumerate(d2v.docvecs.index2entity):
        ax.annotate(name, (proj[i, 0], proj[i, 1]))
    plt.show()


def pipeline(que, ans, pro, tags):
    features = ['questions_title', 'questions_body', 'answers_body',
                'tags_tag_name', 'professionals_industry', 'professionals_headline']
    tp = TextProcessor()
    for df, column in zip([que, que, ans, tags, pro, pro], features):
        df[column] = df[column].apply(tp.process, allow_stopwords=(column == 'tags_tag_name'))

    que_tags = que.merge(tags, left_on='questions_id', right_on='tag_questions_question_id')
    ans_que_tags = ans.merge(que_tags, left_on="answers_question_id", right_on="questions_id")
    df = ans_que_tags.merge(pro, left_on='answers_author_id', right_on='professionals_id')

    d2v = train(df, 'tags_tag_name', features, 10)
    save(d2v, 'tags')

    que_tags = que_tags[['questions_id', 'tags_tag_name']].groupby(by='questions_id', as_index=False) \
        .aggregate(lambda x: ' '.join(x))
    que_tags = que.merge(que_tags, on='questions_id')
    ans_que_tags = ans.merge(que_tags, left_on="answers_question_id", right_on="questions_id")
    df = ans_que_tags.merge(pro, left_on='answers_author_id', right_on='professionals_id')

    d2v = train(df, 'professionals_industry', features, 10)
    save(d2v, 'industries')
