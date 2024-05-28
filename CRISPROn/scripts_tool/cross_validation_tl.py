import os
import numpy as np
import pickle
import keras

from scripts_tool import models_util
from scripts_tool import training_util

gl_init_lr = 0.0008
def create_data(config, DataHandler):
    # Create main dir
    data_dir = f'tool data/datasets/{config.new_data_path}/6_fold/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if os.path.exists(data_dir + '5_fold/'):
        print('Data allready created')
        return


    X = DataHandler['X_train_val']
    dg = DataHandler['dg_train_val']
    y = DataHandler['y_train_val']


    perm = np.random.permutation(X.shape[0])
    val_size = int(X.shape[0] / 6)

    for ind in range(6):
        # Create fold dir
        fold_dir = data_dir + f'{ind}_fold/'
        os.mkdir(fold_dir)

        # Split val train
        valid_ind = perm[ind*val_size:(ind+1)*val_size]

        X_valid = X[valid_ind]
        X_train = np.delete(X, valid_ind, axis=0)

        dg_valid = dg[valid_ind]
        dg_train = np.delete(dg, valid_ind, axis=0)

        y_valid = y[valid_ind]
        y_train = np.delete(y, valid_ind, axis=0)


        pickle.dump(X_valid, open(fold_dir + f'X_valid.pkl', "wb"))
        pickle.dump(X_train, open(fold_dir + f'X_train.pkl', "wb"))
        pickle.dump(dg_valid, open(fold_dir + f'dg_valid.pkl', "wb"))
        pickle.dump(dg_train, open(fold_dir + f'dg_train.pkl', "wb"))
        pickle.dump(y_valid, open(fold_dir + f'y_valid.pkl', "wb"))
        pickle.dump(y_train, open(fold_dir + f'y_train.pkl', "wb"))

def load_fold_data(config, DataHandler, k):
    data_dir =  f'tool data/datasets/{config.new_data_path}/6_fold/'
    data_dir += f'{k}_fold/'


    DataHandler['X_valid'] = pickle.load(open(data_dir + 'X_valid.pkl', "rb"))
    DataHandler['X_train'] = pickle.load(open(data_dir + 'X_train.pkl', "rb"))
    DataHandler['dg_valid'] = pickle.load(open(data_dir + 'dg_valid.pkl', "rb"))
    DataHandler['dg_train'] = pickle.load(open(data_dir + 'dg_train.pkl', "rb"))
    DataHandler['y_valid'] = pickle.load(open(data_dir + 'y_valid.pkl', "rb"))
    DataHandler['y_train'] = pickle.load(open(data_dir + 'y_train.pkl', "rb"))

    return DataHandler

def cross_v_HPS(config, DataHandler):
    create_data(config, DataHandler)
    best_epoch_arr = []
    for k in range(6):
        print(f'\nStarting training {k}')
        keras.backend.clear_session()
        DataHandler = load_fold_data(config, DataHandler, k)
        config.model_num = k+1 #np.random.randint(6) + 1
        if config.train_type == 'gl_tl':
            config.init_lr = gl_init_lr
        model, callback_list = models_util.load_pre_train_model(config, DataHandler)

        history = training_util.train_model(config, DataHandler, model, callback_list)
        best_epoch = int(np.where(history.history['val_loss'] == np.amin(history.history['val_loss']))[0]) + 1
        best_epoch_arr.append(best_epoch)

    opt_epochs = round(np.mean(best_epoch_arr))
    return opt_epochs

def train_6(config, DataHandler):
    DataHandler['X_train'] = np.concatenate((DataHandler['X_train'], DataHandler['X_valid']))
    DataHandler['dg_train'] = np.concatenate((DataHandler['dg_train'], DataHandler['dg_valid']))
    DataHandler['y_train'] = np.concatenate((DataHandler['y_train'], DataHandler['y_valid']))

    for k in range(6):
        print(f'\nStarting training {k}')
        config.model_num = k+1
        if config.train_type == 'gl_tl':
            config.init_lr = gl_init_lr

        model, callback_list = models_util.load_pre_train_model(config, DataHandler)

        history = training_util.train_model(config, DataHandler, model, callback_list)
