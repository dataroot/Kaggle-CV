from pprint import pprint

import pandas as pd
from keras.optimizers import Adam

from doc2vec import pipeline as pipeline_d2v
from processors import QueProc, StuProc, ProProc
from generator import BatchGenerator
from models import Mothership
from evaluation import permutation_importance, plot_fi


def drive(data_path: str, dump_path: str, split_date: str):
    """
    Main function for data preparation, model training and evaluation pipeline
    :param data_path: path to folder with initial .csv data files
    """
    train, test = dict(), dict()

    for var, file_name in [('que', 'questions.csv'), ('ans', 'answers.csv'),
                           ('pro', 'professionals.csv'), ('stu', 'students.csv')]:
        df = pd.read_csv(data_path + file_name)
        col = [col for col in df.columns if 'date' in col][0]

        train[var] = df[df[col] < split_date]
        test[var] = df

        print(var, train[var].shape, test[var].shape)

    tag_que = pd.read_csv(data_path + 'tag_questions.csv')
    tags = pd.read_csv(data_path + 'tags.csv').merge(tag_que, left_on='tags_tag_id', right_on='tag_questions_tag_id')

    # pipeline_d2v(train['que'], train['ans'], train['pro'], tags, 10)

    model = Mothership(que_dim=30, que_input_embs=[102, 42], que_output_embs=[2, 2],
                       pro_dim=25, pro_input_embs=[102, 102, 42], pro_output_embs=[2, 2, 2], inter_dim=10)

    nonneg_pairs = []

    for mode, data in [('Train', train), ('Test', test)]:
        print(mode)

        df = data['que'].merge(data['ans'], left_on='questions_id', right_on='answers_question_id') \
            .merge(data['pro'], left_on='answers_author_id', right_on='professionals_id') \
            .merge(data['stu'], left_on='questions_author_id', right_on='students_id')
        if mode == 'Test':
            df = df.loc[df['answers_date_added'] >= split_date]
        df = df[['questions_id', 'students_id', 'professionals_id']]
        pos_pairs = list(df.itertuples(index=False))
        nonneg_pairs += pos_pairs

        oblige_fit = (mode == 'Train')

        que_proc = QueProc(oblige_fit, dump_path)
        que_data = que_proc.transform(data['que'], tags)
        print('Questions: ', que_data.shape)

        stu_proc = StuProc(oblige_fit, dump_path)
        stu_data = stu_proc.transform(data['stu'], data['que'], data['ans'])
        print('Students: ', stu_data.shape)

        pro_proc = ProProc(oblige_fit, dump_path)
        pro_data = pro_proc.transform(data['pro'], data['que'], data['ans'])
        print('Professionals: ', pro_data.shape)

        print(que_data.columns, stu_data.columns, pro_data.columns)

        bg = BatchGenerator(que_data, stu_data, pro_data, 64, pos_pairs, nonneg_pairs)

        if mode == 'Train':
            model.compile(Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
            model.fit_generator(bg, epochs=3, verbose=2)
            model.save_weights(dump_path + 'model.h5')
        else:
            print(model.evaluate_generator(bg))

        bg = BatchGenerator(que_data, stu_data, pro_data, 2048, pos_pairs, nonneg_pairs)

        fn = {"que": list(stu_data.columns[2:]) + list(que_data.columns[2:]), "pro": list(pro_data.columns[2:]),
              'text': [f'que_emb_{i}' for i in range(10)] + [f'pro_emb_{i}' for i in range(10)]}

        fi = permutation_importance(model, bg[0][0][0], bg[0][0][1], bg[0][1], fn, n_trials=5)
        plot_fi(fi, fn)


if __name__ == '__main__':
    drive('../../data/', 'dump/', '2018-09-01')
