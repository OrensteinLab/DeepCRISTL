import pickle
import numpy as np
from scripts.preprocess import Seq, MultiSeq

def get_data(config):
    print('Reading data')
    if config.transfer_learning:
        dir_path = f'data/{config.pre_train_data}/' # TODO - use tl data
        conf = False
    else:
        dir_path = f'data/pre_train/{config.pre_train_data}/'
        conf = True if config.pre_train_data == 'DeepHF_full' else False

    with open(dir_path + 'test_seq.pkl', "rb") as fp:
        test_seq = pickle.load(fp)

    with open(dir_path + 'valid_seq.pkl', "rb") as fp:
        valid_seq = pickle.load(fp)

    with open(dir_path + 'train_seq.pkl', "rb") as fp:
        train_seq = pickle.load(fp)

    if config.simulation_type == 'full_cross_v': #TODO added this, test
        train_seq = MultiSeq.combine_multi_seqs(train_seq, test_seq)
        



    if config.model_type == 'cnn':
        convert_to_one_hot(test_seq)
        convert_to_one_hot(valid_seq)
        convert_to_one_hot(train_seq)


    DataHandler = {}
    if config.enzyme == 'multi_task':
        enzymes = {'wt': np.array([[0, 0, 1]]), 'esp': np.array([[0, 1, 0]]), 'hf': np.array([[1, 0, 0]])}
        data_dict = {'valid': valid_seq, 'train': train_seq}

        # Adding the train and valid data to data handler serialized
        for key, data in data_dict.items():
            DataHandler[f'X_{key}'] = np.concatenate((data.enzymes_seq['wt'].X,
                                                      data.enzymes_seq['esp'].X,
                                                      data.enzymes_seq['hf'].X), axis=0)

            # Adding the one hot encoded for each enzyme as part of the biofeatures
            for enzyme, ohe in enzymes.items():
                shape = data.enzymes_seq[enzyme].X_biofeat.shape[0]
                ohe_mat = np.repeat(ohe, shape, axis=0)
                data.enzymes_seq[enzyme].X_biofeat = np.concatenate((data.enzymes_seq[enzyme].X_biofeat, ohe_mat),
                                                                    axis=1)

            if config.model_type == 'cnn':
                expand(data)
            DataHandler[f'X_biofeat_{key}'] = np.concatenate((data.enzymes_seq['wt'].X_biofeat,
                                                              data.enzymes_seq['esp'].X_biofeat,
                                                              data.enzymes_seq['hf'].X_biofeat), axis=0)

            DataHandler[f'y_{key}'] = np.concatenate((data.enzymes_seq['wt'].y,
                                                      data.enzymes_seq['esp'].y,
                                                      data.enzymes_seq['hf'].y), axis=0)
            if conf:
                DataHandler[f'conf_{key}'] = np.concatenate((data.enzymes_seq['wt'].confidence,
                                                             data.enzymes_seq['esp'].confidence,
                                                             data.enzymes_seq['hf'].confidence), axis=0)

        # Adding the test data to data handler not serialized - for comparison per enzyme later
        for enzyme, ohe in enzymes.items():
            shape = test_seq.enzymes_seq[enzyme].X_biofeat.shape[0]
            ohe_mat = np.repeat(ohe, shape, axis=0)
            test_seq.enzymes_seq[enzyme].X_biofeat = np.concatenate((test_seq.enzymes_seq[enzyme].X_biofeat, ohe_mat),
                                                                    axis=1)

    else:

        DataHandler['X_train'] = train_seq.enzymes_seq[config.enzyme].X
        DataHandler['X_biofeat_train'] = train_seq.enzymes_seq[config.enzyme].X_biofeat
        DataHandler['y_train'] = train_seq.enzymes_seq[config.enzyme].y

        DataHandler['X_valid'] = valid_seq.enzymes_seq[config.enzyme].X
        DataHandler['X_biofeat_valid'] = valid_seq.enzymes_seq[config.enzyme].X_biofeat
        DataHandler['y_valid'] = valid_seq.enzymes_seq[config.enzyme].y

        if conf:
            DataHandler['conf_train'] = train_seq.enzymes_seq[config.enzyme].confidence
            DataHandler['conf_valid'] = valid_seq.enzymes_seq[config.enzyme].confidence


    if config.model_type == 'cnn':
        expand(test_seq)


    DataHandler['test'] = test_seq

    return DataHandler

def convert_to_one_hot(data_seq):
    enzymes = ['wt', 'esp', 'hf']
    for enzyme in enzymes:
        X = data_seq.enzymes_seq[enzyme].X[:, 1:] - 1
        X = np.eye(4)[X]
        X = X.reshape((X.shape[0], 21, 4, 1))
        data_seq.enzymes_seq[enzyme].X = X


def expand(data_seq):
    enzymes = ['wt', 'esp', 'hf']
    for enzyme in enzymes:
        X_biofeat = data_seq.enzymes_seq[enzyme].X_biofeat
        X_biofeat = X_biofeat.reshape((X_biofeat.shape[0], 14, 1))
        data_seq.enzymes_seq[enzyme].X_biofeat = X_biofeat

def multi_task(test_seq, valid_seq, train_seq, DataHandler, conf, config):
    enzymes = {'wt': np.array([[0, 0, 1]]), 'esp': np.array([[0, 1, 0]]), 'hf': np.array([[1, 0, 0]])}
    data_dict = {'valid': valid_seq, 'train': train_seq}

    # Adding the train and valid data to data handler serialized
    for key, data in data_dict.items():
        DataHandler[f'X_{key}'] = np.concatenate((data.enzymes_seq['wt'].X,
                                                  data.enzymes_seq['esp'].X,
                                                  data.enzymes_seq['hf'].X), axis=0)

        # Adding the one hot encoded for each enzyme as part of the biofeatures
        for enzyme, ohe in enzymes.items():
            shape = data.enzymes_seq[enzyme].X_biofeat.shape[0]
            ohe_mat = np.repeat(ohe, shape, axis = 0)
            data.enzymes_seq[enzyme].X_biofeat = np.concatenate((data.enzymes_seq[enzyme].X_biofeat, ohe_mat), axis=1)

        if config.model_type == 'cnn':
            expand(data)
        DataHandler[f'X_biofeat_{key}'] = np.concatenate((data.enzymes_seq['wt'].X_biofeat,
                                                  data.enzymes_seq['esp'].X_biofeat,
                                                  data.enzymes_seq['hf'].X_biofeat), axis=0)

        DataHandler[f'y_{key}'] = np.concatenate((data.enzymes_seq['wt'].y,
                                                  data.enzymes_seq['esp'].y,
                                                  data.enzymes_seq['hf'].y), axis=0)
        if conf:
            DataHandler[f'conf_{key}'] = np.concatenate((data.enzymes_seq['wt'].confidence,
                                                      data.enzymes_seq['esp'].confidence,
                                                      data.enzymes_seq['hf'].confidence), axis=0)

    # Adding the test data to data handler not serialized - for comparison per enzyme later
    for enzyme, ohe in enzymes.items():
        shape = test_seq.enzymes_seq[enzyme].X_biofeat.shape[0]
        ohe_mat = np.repeat(ohe, shape, axis = 0)
        test_seq.enzymes_seq[enzyme].X_biofeat = np.concatenate((test_seq.enzymes_seq[enzyme].X_biofeat, ohe_mat), axis=1)


