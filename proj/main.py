from keras.optimizers import Adam

from doc2vec import pipeline as pipeline_d2v
from processing import Questions as QueProc, Professionals as ProProc
from generator import BatchGenerator
from modelling import Mothership
from evaluating import *


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

    pipeline_d2v(train['que'], train['ans'], pro, tags, 10)

    model = Mothership(25, [102, 42], [2, 2], 21, [102, 102, 42], [2, 2, 2], 10)
    print(model.summary())

    for mode, data in [('Train', train), ('Test', test)]:
        print(mode)

        qp = QueProc(oblige_fit=(mode == 'Train'))
        qt = qp.transform(data['que'], data['ans'], stu, tags, verbose=False)
        print('Questions: ', qt.shape)

        pp = ProProc(oblige_fit=(mode == 'Train'))
        pt = pp.transform(pro, data['que'], data['ans'], verbose=False)
        print('Professionals: ', pt.shape)

        bg = BatchGenerator(qt, pt, 64)

        if mode == 'Train':
            model.compile(Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
            model.fit_generator(bg, epochs=3, verbose=2)
            model.save_weights('model.h5')
        else:
            print(model.evaluate_generator(bg))

        bg = BatchGenerator(qt, pt, 2048)
        fn = {"que": list(qt.columns[2:]), "pro": list(pt.columns[2:]),
              'text': [f'que_emb_{i}' for i in range(10)] + [f'pro_emb_{i}' for i in range(10)]}
        fi = permutation_importance(model, bg[0][0][0], bg[0][0][1], bg[0][1], fn)
        print(fi)
        plot_fi(fi, fn)


if __name__ == '__main__':
    drive('../../data/', '2018-09-01')
