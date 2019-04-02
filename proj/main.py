from pprint import pprint

import pandas as pd
from keras.optimizers import Adam

from doc2vec import pipeline as pipeline_d2v
from processing import QueProc, ProProc
from generator import BatchGenerator
from modelling import Mothership
from evaluating import permutation_importance, plot_fi


def drive(data_path: str, dump_path: str):
    """
    Main function for data preparation, model training and evaluation pipeline
    :param data_path: path to folder with initial .csv data files
    """
    train, test = dict(), dict()

    # iterate over all the datasets needed to be split in train and test sets
    for var, file_name in [('que', 'questions.csv'), ('ans', 'answers.csv'), ('pro', 'professionals.csv')]:
        df = pd.read_csv(data_path + file_name)

        # find the column to split dataset on
        col = [col for col in df.columns if 'date' in col][0]

        # some examples of possible split
        # if var != 'pro':
        #    df = df[df[col] > '2018-01-01']

        train[var] = df[df[col] < '2018-09-01']
        test[var] = df

        print(var, train[var].shape, test[var].shape)

    # read all the remaining data needed further
    stu = pd.read_csv(data_path + 'students.csv')
    tag_que = pd.read_csv(data_path + 'tag_questions.csv')
    tags = pd.read_csv(data_path + 'tags.csv').merge(tag_que, left_on='tags_tag_id',
                                                     right_on='tag_questions_tag_id')

    # launch pipeline of training and saving to drive updated doc2vec embeddings
    # pipeline_d2v(train['que'], train['ans'], train['pro'], tags, 10)

    # create the main model object
    model = Mothership(que_dim=10, que_input_embs=[], que_output_embs=[],
                       pro_dim=10, pro_input_embs=[], pro_output_embs=[], inter_dim=10)
    # model = Mothership(que_dim=25, que_input_embs=[102, 42], que_output_embs=[2, 2],
    #                   pro_dim=21, pro_input_embs=[102, 102, 42], pro_output_embs=[2, 2, 2], inter_dim=10)
    # print(model.summary())

    # make similar actions for both train and test data
    for mode, data in [('Train', train), ('Test', test)]:
        print(mode)

        # create object for pre-processing question's data
        qp = QueProc(oblige_fit=(mode == 'Train'), path=dump_path)
        # and apply it
        qt = qp.transform(data['que'], data['ans'], stu, tags,
                          # time=None,
                          time=('2018-09-01' if mode == 'Test' else None))
        print('Questions: ', qt.shape)

        '''
        with pd.option_context('display.max_columns', 100):
            pprint(qt.head(10))
        '''

        # same for professionals
        pp = ProProc(oblige_fit=(mode == 'Train'), path=dump_path)
        pt = pp.transform(data['pro'], data['que'], data['ans'])
        print('Professionals: ', pt.shape)

        # create object to generate batches from pre-processed data
        bg = BatchGenerator(qt, pt, 64)

        if mode == 'Train':
            # train and save model in train mode
            model.compile(Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
            model.fit_generator(bg, epochs=10, verbose=2)
            model.save_weights(dump_path + 'model.h5')
        else:
            # evaluate model in test mode
            print(model.evaluate_generator(bg))

        # only to generate one big batch for evaluation purposes
        bg = BatchGenerator(qt, pt, 2048)

        # dict with feature names of both questions and professionals
        # with separated embedding features
        fn = {"que": list(qt.columns[2:]), "pro": list(pt.columns[2:]),
              'text': [f'que_emb_{i}' for i in range(10)] + [f'pro_emb_{i}' for i in range(10)]}

        # calculate feature importance
        fi = permutation_importance(model, bg[0][0][0], bg[0][0][1], bg[0][1], fn, n_trials=5)
        # and plot it
        plot_fi(fi, fn)


if __name__ == '__main__':
    drive('../../data/', 'dump/')
