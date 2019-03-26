import pandas as pd

from doc2vec import pipeline as pipeline_d2v
from processing import Questions as QueProc


def split(df: pd.DataFrame, time: str):
    col = [col for col in df.columns if 'date' in col][0]
    df_before = df[df[col] < time]
    df_after = df[df[col] >= time]
    return df_before, df_after


def drive(data_path: str, time: str):
    train, test = dict(), dict()
    for var, file_name in [('que', 'questions.csv'), ('ans', 'answers.csv')]:
        initial = pd.read_csv(data_path + file_name)
        train[var], test[var] = split(initial, time)
        print(train[var].shape, test[var].shape)

    pro = pd.read_csv(data_path + 'professionals.csv')
    stu = pd.read_csv(data_path + 'students.csv')

    tag_que = pd.read_csv(data_path + 'tag_questions.csv')
    tags = pd.read_csv(data_path + 'tags.csv').merge(tag_que, left_on='tags_tag_id',
                                                     right_on='tag_questions_tag_id')

    # pipeline_d2v(train['que'], train['ans'], pro, tags)

    qp = QueProc(oblige_fit=True)
    qt = qp.transform(train['que'], train['ans'], stu, tags, verbose=False)

    with pd.option_context('display.max_columns', 100):
        print(qt)


drive('../../data/', '2018-06-01')
