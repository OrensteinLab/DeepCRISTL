import pickle
import numpy as np

def get_data(config):
    print('Reading data')
    dir_path = f'tool data/datasets/{config.new_data_path}/'



    with open(dir_path + 'train_val_seq.pkl', "rb") as fp:
        train_val_seq = pickle.load(fp)



    DataHandler = {}

    DataHandler['dg_train_val'] = train_val_seq.dg
    DataHandler['X_train_val'] = train_val_seq.X
    DataHandler['y_train_val'] = train_val_seq.y


def get_user_input_data(sequences):

    DataHandler = {}
    DataHandler['dg_test'] = sequences.dg
    DataHandler['X_test'] = sequences.X
    
    return DataHandler


def get_data_from_dataset(dataset_name):
    dir_path = f'tool data/datasets/{dataset_name}/'
    with open(dir_path + 'train_val_seq.pkl', "rb") as fp:
        test_val_seq = pickle.load(fp)

    DataHandler = {}

    DataHandler['dg_test'] = test_val_seq.dg
    DataHandler['X_test'] = test_val_seq.X
    DataHandler['y_test'] = test_val_seq.y

    return DataHandler