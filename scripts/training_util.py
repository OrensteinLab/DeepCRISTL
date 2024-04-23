import numpy as np
from keras.models import *
from keras.optimizers import *
from sklearn.metrics import mean_squared_error, r2_score
import scipy as sp
from scripts import models_util
from scripts import data_handler as dh
from keras.callbacks import Callback

def scheduler(epoch, lr):
    learning_rate = 0.01  # initial learning rate
    decay_rate = 0.03
    lrate = learning_rate * np.exp(-decay_rate*epoch)
    return lrate


from sklearn.preprocessing import MinMaxScaler, StandardScaler
def debug(config):
    config.em_drop = 0.4
    config.fc_drop = 0.4
    config.batch_size = 110
    config.epochs = 35
    config.em_dim = 68
    config.fc_num_hidden_layers = 1
    config.fc_num_units = 200
    config.fc_activation = 'elu'
    config.optimizer = SGD
    config.cost_function = 'mse'
    config.last_activation = 'linear'
    config.initializer = 'lecun_uniform'
    config.input_scale = None
    config.output_scale = None
    config.rnn_drop = 0.2
    config.rnn_rec_drop = 0.2
    config.rnn_units = 60

    return config



def train_model(config, DataHandler, model, callback_list):
    print('Start training')
    if (config.model_type == 'gl_lstm') and (config.layer_num != 4):
        train_input, y_train = [DataHandler['X_train']], DataHandler['y_train']
        valid_input, y_val = [DataHandler['X_valid']], DataHandler['y_valid']
    else:
        train_input, y_train = [DataHandler['X_train'], DataHandler['X_biofeat_train']], DataHandler['y_train']
        valid_input, y_val = [DataHandler['X_valid'], DataHandler['X_biofeat_valid']], DataHandler['y_valid']

    # Receiving the confidence weights
    if config.pre_train_data == 'DeepHF_full':
        loss_weights = DataHandler['conf_train']
        loss_weights_val = DataHandler['conf_valid']

        weight_scale = np.mean(loss_weights)
        loss_weights = loss_weights / weight_scale
        valid_loss_weights = loss_weights_val / weight_scale

    else:
        loss_weights = None
        valid_loss_weights = None


    # train_input = [train_input[0][:100], train_input[1][:100]]
    # y_train = y_train[:100]
    # loss_weights = loss_weights[:100]
    history = model.fit(train_input,
                        y_train,
                        batch_size=config.batch_size,
                        epochs=config.epochs,
                        verbose=2,
                        validation_data=(valid_input, y_val, valid_loss_weights),
                        shuffle=True,
                        callbacks=callback_list,
                        sample_weight=loss_weights
                        )
    return history




def plot_loss_graph(config):
    # get data
    DataHandler = dh.get_data(config)
    train_input, y_train = [DataHandler['X_train'], DataHandler['X_biofeat_train']], DataHandler['y_train']
    valid_input, y_val = [DataHandler['X_valid'], DataHandler['X_biofeat_valid']], DataHandler['y_valid']

    model, callback_list = models_util.get_model(config, DataHandler)

    history = model.fit(train_input,
                        y_train,
                        batch_size=config.batch_size,
                        epochs=config.epochs,
                        verbose=2,
                        validation_data=(valid_input, y_val),
                        shuffle=True,
                        callbacks=callback_list,
                        )
    
    print(history.history)
    for key in history.history.keys():
        # print key and value
        print(key, history.history[key])

