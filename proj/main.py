import pandas as pd

from doc2vec import pipeline as pipeline_d2v
from processors import QueProc, StuProc, ProProc
from generator import BatchGenerator
from models import DistanceModel, SimpleModel, ConcatModel, Adam
from evaluation import permutation_importance, plot_fi


def drive(data_path: str, dump_path: str, split_date: str):
    """
    Main function for data preparation, model training and evaluation pipeline
    :param data_path: path to folder with initial .csv data files
    :param dump_path: path to all the dump data, like saved models, calculated embeddings etc.
    :param split_date: date used for splitting data on train and test subsets
    """
    train, test = dict(), dict()

    # read and split to train and test all the main datasets
    print('Raw data shapes:')
    for var, file_name in [('que', 'questions.csv'), ('ans', 'answers.csv'),
                           ('pro', 'professionals.csv'), ('stu', 'students.csv')]:
        df = pd.read_csv(data_path + file_name)

        date_col = [col for col in df.columns if 'date' in col][0]
        df[date_col] = pd.to_datetime(df[date_col])

        train[var] = df[df[date_col] < split_date]
        test[var] = df

        print(var, train[var].shape, test[var].shape)

    tags = pd.read_csv(data_path + 'tags.csv')
    tag_que = pd.read_csv(data_path + 'tag_questions.csv') \
        .merge(tags, left_on='tag_questions_tag_id', right_on='tags_tag_id')
    tag_pro = pd.read_csv(data_path + 'tag_users.csv') \
        .merge(tags, left_on='tag_users_tag_id', right_on='tags_tag_id')
    print('tag_que', tag_que.shape)
    print('tag_pro', tag_pro.shape)

    # calculate and save tag and industry embeddings on train data
    # pipeline_d2v(train['que'], train['ans'], train['pro'], tag_que, 10, dump_path)

    nonneg_pairs = []
    for mode, data in [('Train', train), ('Test', test)]:
        print(mode)

        # mappings from professional's id to his registration date. Used in batch generator
        pro_dates = {row['professionals_id']: row['professionals_date_joined'] for i, row in data['pro'].iterrows()}

        # construct dataframe used to extract positive pairs
        df = data['que'].merge(data['ans'], left_on='questions_id', right_on='answers_question_id') \
            .merge(data['pro'], left_on='answers_author_id', right_on='professionals_id') \
            .merge(data['stu'], left_on='questions_author_id', right_on='students_id')
        if mode == 'Test':
            df = df.loc[df['answers_date_added'] >= split_date]
        # else:
        #     df = df.loc[df['answers_date_added'] >= '2016-01-01'] # experiment
        df = df[['questions_id', 'students_id', 'professionals_id', 'answers_date_added']]

        # extract positive pairs, non-negative pairs are all known positive pairs to the moment
        pos_pairs = list(df.itertuples(index=False, name=None))
        nonneg_pairs += pos_pairs
        print(f'Positive pairs number: {len(pos_pairs)}, negative: {len(nonneg_pairs)}')

        # fit new preprocessors in train mode
        oblige_fit = (mode == 'Train')

        # extract and preprocess feature for all three main entities
        # TODO: make this step happen only once for all the data

        que_proc = QueProc(oblige_fit, dump_path)
        que_data = que_proc.transform(data['que'], tag_que)
        print('Questions: ', que_data.shape)

        stu_proc = StuProc(oblige_fit, dump_path)
        stu_data = stu_proc.transform(data['stu'], data['que'], data['ans'])
        print('Students: ', stu_data.shape)

        pro_proc = ProProc(oblige_fit, dump_path)
        pro_data = pro_proc.transform(data['pro'], data['que'], data['ans'], tag_pro)
        print('Professionals: ', pro_data.shape)

        bg = BatchGenerator(que_data, stu_data, pro_data, 64, pos_pairs, nonneg_pairs,
                            que_proc.pp['questions_date_added_time'], pro_dates)
        print('Batches:', len(bg))

        with pd.option_context('display.max_columns', 100, 'display.width', 1024):
            print(stu_data)
            print(que_data)
            print(pro_data)

        if mode == 'Train':
            # in train mode, build, compile train and save model
            model = DistanceModel(que_dim=len(que_data.columns) - 2 + len(stu_data.columns) - 2,
                                  que_input_embs=[102, 42], que_output_embs=[2, 2],
                                  pro_dim=len(pro_data.columns) - 2,
                                  pro_input_embs=[102, 102, 42], pro_output_embs=[2, 2, 2],
                                  inter_dim=16, output_dim=10)

            model.compile(Adam(lr=0.005), loss='binary_crossentropy', metrics=['accuracy'])
            model.fit_generator(bg, epochs=10, verbose=2)

            model.compile(Adam(lr=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
            model.fit_generator(bg, epochs=10, verbose=2)

            model.save_weights(dump_path + 'model.h5')
        else:
            # in test mode just evaluate it
            loss, acc = model.evaluate_generator(bg)
            print(f'Loss: {loss}, accuracy: {acc}')

        # dummy batch generator used to extract single big batch of data to calculate feature importance
        bg = BatchGenerator(que_data, stu_data, pro_data, 1024, pos_pairs, nonneg_pairs,
                            que_proc.pp['questions_date_added_time'], pro_dates)

        # dict with descriptions of feature names, used for visualization of feature importance
        fn = {"que": list(stu_data.columns[2:]) + list(que_data.columns[2:]),  # + ['que_current_time'],
              "pro": list(pro_data.columns[2:]),  # + ['pro_current_time'],
              'text': [f'que_tag_emb_{i}' for i in range(10)] +
                      [f'pro_tag_emb_{i}' for i in range(10)] + [f'pro_ind_emb_{i}' for i in range(10)]}

        print(bg[0][0][0].shape, bg[0][0][1].shape)

        # calculate and plot feature importance
        fi = permutation_importance(model, bg[0][0][0], bg[0][0][1], bg[0][1], fn, n_trials=3)
        plot_fi(fi)


if __name__ == '__main__':
    drive('../../data/', 'dump/', '2019-01-01')
