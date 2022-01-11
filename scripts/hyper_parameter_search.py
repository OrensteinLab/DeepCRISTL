import numpy as np
from keras.optimizers import *
import os
import pandas as pd
from scripts import models_util
from scripts import training_util
import scipy as sp


# Global dictionaries
initializer_dict = {'0': 'he_uniform', '1': 'lecun_uniform', '2': 'normal', '3': 'he_normal'}
optimizer_dict = {'0': Nadam, '1': SGD, '2': RMSprop, '3': Adagrad, '4': Adadelta, '5': Adam, '6': Adamax}
fc_activation_dict = {'0': 'elu', '1': 'relu', '2': 'tanh', '3': 'sigmoid', '4': 'hard_sigmoid'}
last_activation_dict = {'0': 'sigmoid', '1': 'linear'}

def get_bounds(config):


    basic_bounds = {
        'last_activation': np.array(tuple(map(int, tuple(last_activation_dict.keys())))),
        'initializer': np.array(tuple(map(int, tuple(initializer_dict.keys())))),
        'batch_size': np.arange(start=30, stop=200, step=10),
        'optimizer': np.array(tuple(map(int, tuple(optimizer_dict.keys())))),
    }

    embedding_bounds = {
        'em_dim': np.arange(start=30, stop=80, step=1),
        'em_drop': np.arange(start=0.1, stop=1.0, step=0.1)
    }

    rnn_bounds = {
        'rnn_units': np.arange(start=50, stop=230, step=10),
        'rnn_drop': np.arange(start=0.1, stop=1.0, step=0.1),
        'rnn_rec_drop': np.arange(start=0.1, stop=1.0, step=0.1),
        # 'rnn_l2': np.arange(start=0.01, stop=0.5, step=0.01),
        # 'rnn_rec_l2': np.arange(start=0.01, stop=0.5, step=0.01)
    }

    fc_bounds = {
        'fc_num_hidden_layers': np.arange(start=1, stop=5, step=1),
        'fc_num_units': np.arange(start=50, stop=310, step=10),
        'fc_activation': np.array(tuple(map(int, tuple(fc_activation_dict.keys())))),
        # 'fc_l2': np.arange(start=0.01, stop=0.5, step=0.01),
        'fc_drop': np.arange(start=0.1, stop=1.0, step=0.1)
    }

    if config.model_type == 'lstm':
        bounds = {**basic_bounds, **embedding_bounds, **rnn_bounds, **fc_bounds}

    return bounds


def get_dir(config, bounds):
    dir_path = 'HPS/'
    if config.transfer_learning:
        a=0 # TODO
    else:
        dir_path += 'pre_train/'

    dir_path += f'{config.pre_train_data}/{config.enzyme}'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    df_path = dir_path + '/HPS_0.csv'
    ind = 0
    while os.path.exists(df_path):
        ind += 1
        df_path = dir_path + f'/HPS_{ind}.csv'

    df = create_df(config, bounds)
    print(f'df_path={df_path}')
    df.to_csv(df_path)
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

    setattr(config, 'initializer', initializer_dict[str(config.initializer)])
    setattr(config, 'optimizer', optimizer_dict[str(config.optimizer)])
    setattr(config, 'fc_activation', fc_activation_dict[str(config.fc_activation)])
    setattr(config, 'last_activation', last_activation_dict[str(config.last_activation)])


def save_results(config, history, bounds, df_path, df, val_spearman):
    new_line = {}
    for key in bounds.keys():
        val = getattr(config, key)
        new_line[key] = val

    best_loss = np.min(history.history['val_loss'])
    new_line['loss'] = best_loss

    if config.enzyme == 'multi_task':
        enzymes = ['wt', 'esp', 'hf']
        mean_spearman = 0

        for enzyme in enzymes:
            mean_spearman += val_spearman[enzyme]
            new_line[f'{enzyme} spearman'] = val_spearman[enzyme]
        mean_spearman /= 3
        new_line['mean spearman'] = mean_spearman

    else:
        new_line['spearman'] = val_spearman

    df = df.append(new_line, ignore_index=True)
    df.to_csv(df_path)
    return df


def get_val_spearman(config, model, DataHandler):
    val_spearman = {}

    if config.enzyme == 'multi_task':
        enzymes = {'wt': -1, 'esp': -2, 'hf': -3}
        for enzyme, ohe in enzymes.items():
            inputs_index = np.where(DataHandler['X_biofeat_valid'][:,ohe] ==  1)[0]
            inputs = [DataHandler['X_valid'][inputs_index, :], DataHandler['X_biofeat_valid'][inputs_index, :]]
            true_labels = DataHandler['y_valid'][inputs_index]
            predictions = model.predict(inputs)
            predictions = np.squeeze(predictions, axis=1)
            spearman = sp.stats.spearmanr(true_labels, predictions)[0]
            val_spearman[enzyme] = spearman
            print(f'Enzyme: {enzyme}, Spearman: {spearman}')

    else:
        inputs = [DataHandler['X_valid'], DataHandler['X_biofeat_valid']]
        true_labels = DataHandler['y_valid']
        predictions = model.predict(inputs)
        predictions = np.squeeze(predictions, axis=1)
        val_spearman = sp.stats.spearmanr(true_labels, predictions)[0]
        print(f'Enzyme: {config.enzyme}, Spearman: {val_spearman}')

    return val_spearman

def param_search(config, DataHandler):
    bounds = get_bounds(config)
    df_path, df = get_dir(config, bounds)
    config.epochs = 50
    for sim_ind in range(10):
        generate_parameters(config, bounds)
        model, callback_list = models_util.get_model(config, DataHandler)
        history = training_util.train_model(config, DataHandler, model, callback_list)
        val_spearman = get_val_spearman(config, model, DataHandler)
        df = save_results(config, history, bounds, df_path, df, val_spearman)
