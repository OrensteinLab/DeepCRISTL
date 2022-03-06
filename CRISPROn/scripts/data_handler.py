import pickle
import numpy as np

def get_data(config, set):
    print('Reading data')
    dir_path = f'data/tl_train/{config.tl_data_category}/{config.tl_data}/set{set}/'



    with open(dir_path + 'test_seq.pkl', "rb") as fp:
        test_seq = pickle.load(fp)

    with open(dir_path + 'valid_seq.pkl', "rb") as fp:
        valid_seq = pickle.load(fp)

    with open(dir_path + 'train_seq.pkl', "rb") as fp:
        train_seq = pickle.load(fp)


    DataHandler = {}

    DataHandler['dg_train'] = train_seq.dg
    DataHandler['dg_valid'] = valid_seq.dg
    DataHandler['dg_test'] = test_seq.dg

    DataHandler['X_train'] = train_seq.X
    DataHandler['y_train'] = train_seq.y

    DataHandler['X_valid'] = valid_seq.X
    DataHandler['y_valid'] = valid_seq.y

    DataHandler['X_test'] = test_seq.X
    DataHandler['y_test'] = test_seq.y


    return DataHandler