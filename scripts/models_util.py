import keras
from keras.preprocessing import text
from keras.preprocessing import sequence
from keras.layers import Embedding, Bidirectional, concatenate, Conv2D, BatchNormalization, MaxPool2D
from keras.layers import *
from keras.models import *
from keras.callbacks import EarlyStopping
import numpy as np
from keras.callbacks import Callback
import os
from keras import backend as K

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
        self.patience = 5
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
                    self.patience = 5
                    # self.model.save(filepath, overwrite=True)
                else:
                    if self.lr_scheduler:
                        self.patience -= 1
                        if self.patience == 0:
                            self.patience = 5
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


def lstm_model(config, DataHandler):

    # Lstm input layer
    seq_input_shape = DataHandler['X_train'].shape[1]
    sequence_input = Input(name='seq_input', shape=(seq_input_shape,))

    # Embedding layer
    embedding_layer = Embedding(5, config.em_dim, input_length=seq_input_shape)
    embedded = embedding_layer(sequence_input)
    embedded = SpatialDropout1D(config.em_drop)(embedded)
    x = embedded

    # RNN
    lstm = LSTM(config.rnn_units, dropout=config.rnn_drop,
                #kernel_regularizer=L2(l2=config.rnn_l2), recurrent_regularizer=L2(l2=config.rnn_rec_l2),
                kernel_regularizer='l2', recurrent_regularizer='l2',
                recurrent_dropout=config.rnn_rec_drop, return_sequences=True, kernel_initializer=config.initializer)
    x = Bidirectional(lstm)(x)
    x = Flatten()(x)

    # Biological features
    bio_input_shape = DataHandler['X_biofeat_train'].shape[1]
    biological_input = Input(name = 'bio_input', shape = (bio_input_shape,))
    x = concatenate([x, biological_input])

    # Fully connected layers
    for l in range(config.fc_num_hidden_layers):
        x = Dense(config.fc_num_units, activation=config.fc_activation, kernel_initializer=config.initializer)(x)
        x = Dropout(config.fc_drop)(x)


    # Output layer
    output = Dense(1, activation=config.last_activation, name='output', kernel_initializer=config.initializer)(x)

    # Defining model
    model = Model(inputs=[sequence_input, biological_input], outputs=[output])
    model.compile(loss='mse', optimizer=config.optimizer(lr=0.002))
    model.summary()


    # Callbacks
    callback_list = []
    early_stopping = EarlyStopping(monitor='val_loss', patience=8, verbose=1)
    callback_list.append(early_stopping)

    if config.save_model:
        save_model_path = get_save_path(config)
    else:
        save_model_path = None

    get_best_model = GetBest(filepath=save_model_path , verbose=1, save_best=config.save_model)
    callback_list.append(get_best_model)

    return model, callback_list


def gl_lstm_model(config, DataHandler):



    if config.layer_num == 1:
        # Lstm input layer
        seq_input_shape = DataHandler['X_train'].shape[1]
        sequence_input = Input(name='seq_input', shape=(seq_input_shape,))
        # Embedding layer
        embedding_layer = Embedding(5, config.em_dim, input_length=seq_input_shape)
        embedded = embedding_layer(sequence_input)
        embedded = SpatialDropout1D(config.em_drop)(embedded)
        x = embedded

        # RNN
        lstm = LSTM(config.rnn_units, dropout=config.rnn_drop,
                    #kernel_regularizer=L2(l2=config.rnn_l2), recurrent_regularizer=L2(l2=config.rnn_rec_l2),
                    kernel_regularizer='l2', recurrent_regularizer='l2',
                    recurrent_dropout=config.rnn_rec_drop, return_sequences=True, kernel_initializer=config.initializer)
        x = Bidirectional(lstm)(x)
        x = Flatten()(x)

    else:
        model_path = f'models/pre_train/{config.pre_train_data}/{config.enzyme}/gl_lstm/model_L{config.layer_num - 1}_0/model'
        pre_model = keras.models.load_model(model_path)

        sequence_input = pre_model.inputs[0]
        x = pre_model.layers[1 + config.layer_num].output

        # RNN
        lstm = LSTM(config.rnn_units, dropout=config.rnn_drop,
                    #kernel_regularizer=L2(l2=config.rnn_l2), recurrent_regularizer=L2(l2=config.rnn_rec_l2),
                    kernel_regularizer='l2', recurrent_regularizer='l2',
                    recurrent_dropout=config.rnn_rec_drop, return_sequences=True, kernel_initializer=config.initializer)
        x = Bidirectional(lstm, name=f'bidirectional_{config.layer_num}')(x)

        x = Flatten()(x)

    # Fully connected layers
    x = Dense(config.fc_num_units, activation=config.fc_activation, kernel_initializer=config.initializer)(x)
    x = Dropout(config.fc_drop)(x)


    # Output layer
    output = Dense(1, activation=config.last_activation, name='output', kernel_initializer=config.initializer)(x)

    # Defining model
    model = Model(inputs=[sequence_input], outputs=[output])
    model.compile(loss='mse', optimizer=config.optimizer(lr=0.002))

    if config.layer_num != 1:
        model.set_weights(pre_model.get_weights()[0:2 + config.layer_num])
        for layer in model.layers[0:2 + config.layer_num]:
            layer.trainable = False
    model.summary()


    # Callbacks
    callback_list = []
    early_stopping = EarlyStopping(monitor='val_loss', patience=8, verbose=1)
    callback_list.append(early_stopping)

    if config.save_model:
        save_model_path = get_save_path(config)
    else:
        save_model_path = None

    get_best_model = GetBest(filepath=save_model_path , verbose=1, save_best=config.save_model)
    callback_list.append(get_best_model)

    return model, callback_list



def get_model(config, DataHandler):
    if config.model_type == 'lstm':
        return lstm_model(config, DataHandler)

    if config.model_type == 'cnn':
        a=0 # TODO
        # return cnn_model(config, DataHandler)

    if config.model_type == 'gl_lstm':
        return gl_lstm_model(config, DataHandler)


def load_pre_train_model(config, DataHandler, verbose=1):
    if config.train_type == 'gl_tl':
        model_path = f'models/transfer_learning/{config.tl_data}/set{config.set}/{config.pre_train_data}/{config.enzyme}/LL_tl/model_{config.model_num}/model'
    else:
        model_path = f'models/pre_train/{config.pre_train_data}/{config.enzyme}/full_cross_v/model_{config.model_num}/model'
    if config.flanks:
        model_path = f'models/transfer_learning/{config.tl_data}/{config.pre_train_data}/{config.enzyme}/no_em_tl/model_{config.model_num}/model'
    model = load_model(model_path)

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

    if config.flanks or config.new_features:
        model = upgrade_model(config, model, DataHandler)
        model.compile(loss='mse', optimizer=config.optimizer(lr=config.init_lr))
        callback_list[1] = GetBest(filepath=save_model_path , verbose=1, save_best=config.save_model, init_lr=config.init_lr, lr_scheduler=True)

    if verbose > 0:
        model.summary()
    return model, callback_list

# Utils
def get_save_path(config):
    path = 'models/'
    if not os.path.exists(path):
        os.mkdir(path)

    if config.transfer_learning:
        path += f'transfer_learning/{config.tl_data}/set{config.set}/{config.pre_train_data}/{config.enzyme}/{config.train_type}/'
        if not os.path.exists(path):
            os.makedirs(path)
    else:
        path += f'pre_train/{config.pre_train_data}/{config.enzyme}/'
        if config.model_type != 'lstm':
            path += f'{config.model_type}/'
        if config.simulation_type == 'full_cross_v':
            path += 'full_cross_v/'
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


# Add flanks or new_features for transfer learning
def upgrade_model(config, base_model, DataHandler):
    # base_model.trainable = False
    # for layer in base_model.layers:
    #     layer.trainable = False
    if config.flanks:
        # up flank
        up_input_shape = DataHandler['up_train'].shape[1:]
        up_input = Input(name='up_input', shape=up_input_shape)
        up7 = Conv2D(40, (7,4), input_shape=up_input_shape)(up_input)
        up_pool7 = MaxPool2D(pool_size=(2, 1))(up7)
        up_flat7 = Flatten(name='up7_flatten')(up_pool7)

        up5 = Conv2D(70, (5,4), input_shape=up_input_shape)(up_input)
        up_pool5 = MaxPool2D(pool_size=(2, 1))(up5)
        up_flat5 = Flatten(name='up5_flatten')(up_pool5)

        up3 = Conv2D(100, (3,4), input_shape=up_input_shape)(up_input)
        up_pool3 = MaxPool2D(pool_size=(2, 1))(up3)
        up_flat3 = Flatten(name='up3_flatten')(up_pool3)

        up_concat = concatenate([up_flat7, up_flat5, up_flat3], name='up_concat')
        up_dense = Dense(80, name='up_dense')(up_concat)
        up_out = Dropout(0.3, name='up_dropout')(up_dense)
        up_dense = Dense(80, name='up_dense2')(up_out)
        up_out = Dropout(0.3, name='up_dropout2')(up_dense)
        # up_out = Dense(1, name='up_dense_out')(up_out)

        # down flank
        down_input_shape = DataHandler['down_train'].shape[1:]
        down_input = Input(name='down_input', shape=down_input_shape)
        down7 = Conv2D(40, (7,4), input_shape=down_input_shape)(down_input)
        down_pool7 = MaxPool2D(pool_size=(2, 1))(down7)
        down_flat7 = Flatten(name='down7_flatten')(down_pool7)

        down5 = Conv2D(70, (5,4), input_shape=down_input_shape)(down_input)
        down_pool5 = MaxPool2D(pool_size=(2, 1))(down5)
        down_flat5 = Flatten(name='down5_flatten')(down_pool5)

        down3 = Conv2D(100, (3,4), input_shape=down_input_shape)(down_input)
        down_pool3 = MaxPool2D(pool_size=(2, 1))(down3)
        down_flat3 = Flatten(name='down3_flatten')(down_pool3)

        down_concat = concatenate([down_flat7, down_flat5, down_flat3], name='down_concat')
        down_dense = Dense(80, name='down_dense')(down_concat)
        down_out = Dropout(0.3, name='down_dropout')(down_dense)
        down_dense = Dense(80, name='down_dense2')(down_out)
        down_out = Dropout(0.3, name='down_dropout2')(down_dense)
        # down_out = Dense(1, name='down_dense_out')(down_out)

    if config.new_features:
        # new features
        new_features_input_shape = DataHandler['new_features_train'].shape[1:]
        new_features_input = Input(name='new_features_input', shape=new_features_input_shape)
        new_features_dense = Dense(100, name='new_features_dense')(new_features_input)
        new_features_out = Dropout(0.3, name='new_features_dropout')(new_features_dense)



    # final model
    out_layers = [base_model.layers[-3].output]
    inputs = base_model.inputs
    if config.flanks:
        out_layers += [up_out, down_out]
        inputs += [up_input, down_input]

    if config.new_features:
        out_layers += [new_features_out]
        inputs += [new_features_input]

    model_out = concatenate(out_layers, name='full_concat')
    model_out = Dense(100, name='full_dense1')(model_out)
    model_out = Dropout(0.3, name='full_dropout1')(model_out)
    model_out = Dense(1, name='model_out')(model_out)

    model = Model(inputs=inputs, outputs=[model_out])

    return model

