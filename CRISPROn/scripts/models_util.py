from keras.models import load_model
import tensorflow as tf
from keras.callbacks import EarlyStopping, Callback
import warnings
import os
import numpy as np
from keras import backend as K
import keras


class GetBest(Callback):
    def __init__(self, filepath=None, monitor='val_loss', save_best=False, verbose=0,
                 mode='auto', period=1, val_data=None, init_lr=0.002, lr_scheduler=True):
        super(GetBest, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.period = period
        self.save_best = save_best
        self.filepath = filepath
        self.best_epochs = 0
        self.epochs_since_last_save = 0
        self.val_data = val_data
        self.patience = 50
        self.init_lr = init_lr
        self.lr = self.init_lr
        self.lr_scheduler = lr_scheduler

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('GetBest mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_train_begin(self, logs=None):
        self.best_weights = self.model.get_weights()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            # filepath = self.filepath.format(epoch=epoch + 1, **logs)
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn('Can pick best model only with %s available, '
                              'skipping.' % (self.monitor), RuntimeWarning)
            else:
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                              ' storing weights.'
                              % (epoch + 1, self.monitor, self.best,
                                 current))
                    self.best = current
                    self.best_epochs = epoch + 1
                    self.best_weights = self.model.get_weights()
                    self.patience = 50
                    # self.model.save(filepath, overwrite=True)
                else:
                    if self.lr_scheduler:
                        self.patience -= 1
                        if self.patience == 0:
                            self.patience = 50
                            self.lr /= 2
                            # self.model.optimizer.lr.assign(self.lr)
                            K.set_value(self.model.optimizer.lr, self.lr)
                            print(f'updating learning rate: {self.lr * 2} -> {self.lr}')
                            # self.model.set_weights(self.best_weights) #TODO - remove this line
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve.' %
                                  (epoch + 1, self.monitor))
                            print(f'patience: {self.patience}')


    def on_train_end(self, logs=None):
        if self.verbose > 0:
            print('Using epoch %05d with %s: %0.5f.' % (self.best_epochs, self.monitor,
                                                        self.best))
        self.model.set_weights(self.best_weights)
        if self.save_best:
            self.model.save(self.filepath, overwrite=True)



def load_pre_train_model(config, DataHandler, verbose=1):
    if config.train_type == 'gl_tl':
        model_path = f'tl_models/transfer_learning/{config.tl_data}/set{config.set}/LL_tl/model_{config.model_num-1}/model'
    else:
        model_path = f'data/deep_models/best/{config.model_num}.model.best/'


    model = tf.keras.models.load_model(model_path)


    # Callbacks
    callback_list = []
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=verbose)
    callback_list.append(early_stopping)

    if config.save_model:
        save_model_path = get_save_path(config)
    else:
        save_model_path = None
    get_best_model = GetBest(filepath=save_model_path , verbose=verbose, save_best=config.save_model, init_lr=config.init_lr, lr_scheduler=False)
    callback_list.append(get_best_model)

    # if set_params:
    #     model.layers[2].rate = config.em_drop
    #     for layer in model.layers[3:]:
    #         if 'dropout' in layer.name:
    #             layer.rate = config.fc_drop
    weights = model.get_weights()
    model = keras.models.clone_model(model)
    if config.train_type == 'LL_tl':
        for layer in model.layers[:-3]:
            layer.trainable = False
    if config.train_type == 'gl_tl':
        for layer in model.layers[2:]:
            layer.trainable = True
    if config.train_type == 'no_em_tl':
        model.layers[1].trainable = False

    model.compile(loss='mse', optimizer=config.optimizer(lr=config.init_lr))
    if config.train_type != 'no_pre_train':
        model.set_weights(weights)



    if verbose > 0:
        model.summary()
    return model, callback_list


# Utils
def get_save_path(config):
    path = 'tl_models/'
    if not os.path.exists(path):
        os.mkdir(path)

    path += f'transfer_learning/{config.tl_data}/set{config.set}/{config.train_type}/'
    if not os.path.exists(path):
        os.makedirs(path)



    ind = 0
    model_path = path + 'model_0/'
    while os.path.exists(model_path):
        ind += 1
        model_path = f'{path}model_{ind}/'
    os.mkdir(model_path)
    model_path += 'model'

    return model_path