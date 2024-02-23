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


    return DataHandler