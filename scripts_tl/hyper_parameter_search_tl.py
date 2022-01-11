import numpy as np
from keras.optimizers import *
import os
import pandas as pd
from scripts import models_util
from scripts_tl import training_util_tl
import scipy as sp
import keras


# Global dictionaries
name_to_optimizer_dict = {'Nadam': Nadam, 'SGD': SGD, 'RMSprop': RMSprop, 'Adagrad': Adagrad, 'Adadelta': Adadelta, 'Adam': Adam, 'Adamax': Adamax}
initializer_dict = {'0': 'he_uniform', '1': 'lecun_uniform', '2': 'normal', '3': 'he_normal'}
optimizer_dict = {'0': Nadam, '1': SGD, '2': RMSprop, '3': Adagrad, '4': Adadelta, '5': Adam, '6': Adamax}

fc_activation_dict = {'0': 'elu', '1': 'relu', '2': 'tanh', '3': 'sigmoid', '4': 'hard_sigmoid'}
last_activation_dict = {'0': 'sigmoid', '1': 'linear'}

def get_bounds(config):


    basic_bounds = {
        # 'last_activation': np.array(tuple(map(int, tuple(last_activation_dict.keys())))),
        'batch_size': np.arange(start=30, stop=200, step=10),
        'optimizer': np.array(tuple(map(int, tuple(optimizer_dict.keys())))),
        'init_lr': np.arange(start=0.001, stop=0.05, step=0.001),

    }

    # embedding_bounds = {
    #     'em_drop': np.arange(start=0.1, stop=1.0, step=0.1)
    # }

    # rnn_bounds = {
    #     'rnn_drop': np.arange(start=0.1, stop=1.0, step=0.1),
    #     'rnn_rec_drop': np.arange(start=0.1, stop=1.0, step=0.1),
    #     # 'rnn_l2': np.arange(start=0.01, stop=0.5, step=0.01),
    #     # 'rnn_rec_l2': np.arange(start=0.01, stop=0.5, step=0.01)
    # }

    # fc_bounds = {
    #     # 'fc_activation': np.array(tuple(map(int, tuple(fc_activation_dict.keys())))),
    #     # 'fc_l2': np.arange(start=0.01, stop=0.5, step=0.01),
    #     'fc_drop': np.arange(start=0.1, stop=1.0, step=0.1)
    # }

    if config.model_type == 'lstm':
        bounds = {**basic_bounds}

    return bounds


def get_dir(config, bounds):
    dir_path = f'HPS/transfer_learning/{config.tl_data_category}/{config.tl_data}/{config.enzyme}/{config.train_type}'

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    df_path = dir_path + '/HPS_0.csv'
    ind = 0
    while os.path.exists(df_path):
        ind += 1
        df_path = dir_path + f'/HPS_{ind}.csv'

    df = create_df(config, bounds)
    print(f'df_path={df_path}')
    df.to_csv(df_path, index=False)
    return df_path, df

def create_df(config, bounds):
    if config.enzyme == 'multi_task':
        columns = ['wt spearman', 'esp spearman', 'hf spearman', 'mean spearman', 'loss']
    else:
        columns = ['spearman', 'loss']

    for key in bounds.keys():
        columns.append(key)
    df = pd.DataFrame(columns = columns)
    return df



def generate_parameters(config, bounds):
    for param, rng in bounds.items():
        setattr(config, param, np.random.choice(rng))

    setattr(config, 'optimizer', optimizer_dict[str(config.optimizer)])
    # setattr(config, 'fc_activation', fc_activation_dict[str(config.fc_activation)])
    # setattr(config, 'last_activation', last_activation_dict[str(config.last_activation)])


def save_results(config, history, bounds, df_path, df, val_spearman, sim_ind):
    new_line = {}
    for key in bounds.keys():
        val = getattr(config, key)
        if key == 'optimizer':
            val = str(val).split('.')[-1].split('\'')[0]
        new_line[key] = val

    best_loss = np.min(history.history['val_loss'])
    new_line['loss'] = best_loss
    new_line['spearman'] = val_spearman

    print(f'Simulation: {sim_ind}, Spearman: {val_spearman}, Loss: {best_loss}')
    if np.isnan(best_loss):
        print('Loss is nan -> returning')
        return df # Dont use this parameters

    df = df.append(new_line, ignore_index=True)
    df.to_csv(df_path)
    return df


def get_val_spearman(config, model, DataHandler):
    val_spearman = {}


    inputs = [DataHandler['X_valid'], DataHandler['X_biofeat_valid']]
    true_labels = DataHandler['y_valid']
    predictions = model.predict(inputs)
    predictions = np.squeeze(predictions, axis=1)
    val_spearman = sp.stats.spearmanr(true_labels, predictions)[0]
    print(f'Spearman: {val_spearman}')

    return val_spearman

def param_search(config, DataHandler):
    bounds = get_bounds(config)
    df_path, df = get_dir(config, bounds)
    config.epochs = 500
    for sim_ind in range(50):
        print(f'\nStarting HPS number {sim_ind}')
        generate_parameters(config, bounds)
        model, callback_list = models_util.load_pre_train_model(config)
        history = training_util_tl.train_model(config, DataHandler, model, callback_list)
        val_spearman = get_val_spearman(config, model, DataHandler)
        df = save_results(config, history, bounds, df_path, df, val_spearman)


def tl_param_search(config, DataHandler):
    bounds = get_bounds(config)
    df_path, df = get_dir(config, bounds)
    config.epochs = 200
    config.model_num = 2
    for sim_ind in range(100):
        generate_parameters(config, bounds)
        model, callback_list = models_util.load_pre_train_model(config, DataHandler, verbose=0)
        history = training_util_tl.train_model(config, DataHandler, model, callback_list, verbose=0)
        val_spearman = get_val_spearman(config, model, DataHandler)
        df = save_results(config, history, bounds, df_path, df, val_spearman, sim_ind)
        keras.backend.clear_session()

    best_params = df.iloc[df[['spearman']].idxmax()]
    config.batch_size = best_params['batch_size'].values[0]
    config.init_lr = best_params['init_lr'].values[0]
    config.optimizer = name_to_optimizer_dict[best_params['optimizer'].values[0]]
