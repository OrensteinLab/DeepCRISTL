
import numpy as np
from keras.models import *
from keras.optimizers import *
from sklearn.metrics import mean_squared_error, r2_score
import scipy as sp
from scripts import models_util
from scripts import data_handler as dh
from keras.callbacks import Callback
from scripts_tl import data_handler_tl as dh_tl
from scripts_tl import configurations_tl as cfg_tl
from scripts_tl import cross_validation_tl as cv_tl



def train_model(config, DataHandler, model, callback_list, verbose=2):
    if verbose > 0:
        print('Start training')
    train_input, y_train = [DataHandler['X_train'], DataHandler['X_biofeat_train']], DataHandler['y_train']
    valid_input, y_val = [DataHandler['X_valid'], DataHandler['X_biofeat_valid']], DataHandler['y_valid']

    if config.flanks:
        train_input += [DataHandler['up_train'], DataHandler['down_train']]
        valid_input += [DataHandler['up_valid'], DataHandler['down_valid']]

    if config.new_features:
        train_input += [DataHandler['new_features_train']]
        valid_input += [DataHandler['new_features_valid']]


    history = model.fit(train_input,
                        y_train,
                        batch_size=config.batch_size,
                        epochs=config.epochs,
                        verbose=verbose,
                        validation_data=(valid_input, y_val),
                        shuffle=True,
                        callbacks=callback_list,
                        )
    return history




def plot_loss_graph(config):
    config.tl_data = 'morenoMateos2015'
    config.set = 0
    config.save_model = False
    config.epochs = 1000


    train_types = ['full_tl']
    for train_type in train_types:
        config.train_type = train_type
        DataHandler = dh_tl.get_data(config, config.set)
        opt_epochs = cv_tl.cross_v_HPS(config, DataHandler)
        config.epochs = opt_epochs
        # get data
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
        
        for key in history.history.keys():
            # print key and value
            print(key, history.history[key])