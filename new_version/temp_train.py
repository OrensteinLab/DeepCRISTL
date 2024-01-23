import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import models
from keras import optimizers
from keras import losses
import wandb

import numpy as np
import matplotlib.pyplot as plt

import os
import pandas as pd

from scipy.stats import spearmanr

ENZYME = 'wt'
SEQ_INPUT_SHAPE = (21*4)
BIO_INPUT_SHAPE = 11



def init_wandb(config):
    wandb.init(project='deepCRISTL', config=config)
    config = wandb.config
    config.update({'version': "1.0"})

    return 

def get_sequence_features_and_labels(file, config):
    df = pd.read_csv(file)

    # remove all lines with NaN values
    df = df.dropna()

    sequences = df['21mer'].values
    encoded_sequences = one_hot_encode_array(sequences)
    # flatten
    encoded_sequences = encoded_sequences.reshape(encoded_sequences.shape[0], SEQ_INPUT_SHAPE)

    # epi features are named epi1 epi2 epi3 upt ot epi11
    epi_features = ['epi' + str(i) for i in range(1, 12)]
    bio_features = df[epi_features].values
    bio_features = bio_features.astype(np.float32)
    bio_features = np.array(bio_features)

    label_name = config['enzyme'] + '_mean_eff'
    labels = df[label_name].values
    labels = labels.astype(np.float32)
    labels = np.array(labels)

    # save all to files that are readable by human
    #np.savetxt('encoded_sequences_{}.txt'.format(file), encoded_sequences, fmt='%s')
    #np.savetxt('bio_features_{}.txt'.format(file), bio_features, fmt='%s')
    #np.savetxt('labels_{}.txt'.format(file), labels, fmt='%s')


    return encoded_sequences, bio_features, labels


def one_hot_encode_sequence(sequence):
    # create a dictionary for encoding
    encoding = {'A': [1, 0, 0, 0],
                'C': [0, 1, 0, 0],
                'G': [0, 0, 1, 0],
                'T': [0, 0, 0, 1],
                'N': [0, 0, 0, 0]}

    # encode the sequence
    encoded_sequence = [encoding[base] for base in sequence]
    # make into numpy array
    encoded_sequence = np.array(encoded_sequence)

    return encoded_sequence


def one_hot_encode_array(array):
    return np.array([one_hot_encode_sequence(sequence) for sequence in array])


def get_config_for_enzyme(enzyme):
    config = {}  # Initialize an empty dictionary

    #update the enzyme 
    config.update({'enzyme': enzyme})

    if enzyme == 'multi_task':
        config.update({
            'em_dim': 43,
            'em_drop': 0.1,
            'rnn_drop': 0.1,
            'rnn_rec_drop': 0.5,
            'rnn_units': 220,
            'fc_num_hidden_layers': 3,
            'fc_num_units': 190,
            'fc_drop': 0.4,
            'fc_activation': 'elu',
            'batch_size': 130,
            'epochs': 100,
            'optimizer': keras.optimizers.Adamax,
            'last_activation': 'sigmoid',
            'initializer': 'he_uniform'
        })
    elif enzyme == 'esp':
        config.update({
            'em_dim': 36,
            'em_drop': 0.5,
            'rnn_drop': 0.4,
            'rnn_rec_drop': 0.4,
            'rnn_units': 130,
            'fc_num_hidden_layers': 1,
            'fc_num_units': 90,
            'fc_drop': 0.4,
            'fc_activation': 'relu',
            'batch_size': 100,
            'epochs': 100,
            'optimizer': keras.optimizers.RMSprop,
            'last_activation': 'sigmoid',
            'initializer': 'he_uniform'
        })
    elif enzyme == 'wt':
        config.update({
            'em_dim': 63,
            'em_drop': 0.8,
            'rnn_drop': 0.1,
            'rnn_rec_drop': 0.2,
            'rnn_units': 110,
            'fc_num_hidden_layers': 3,
            'fc_num_units': 220,
            'fc_drop': 0.6,
            'fc_activation': 'sigmoid',
            'batch_size': 90,
            'epochs': 100,
            'optimizer': keras.optimizers.Adamax,
            'last_activation': 'linear',
            'initializer': 'normal'
        })
    elif enzyme == 'hf':
        config.update({
            'em_dim': 67,
            'em_drop': 0.1,
            'rnn_drop': 0.3,
            'rnn_rec_drop': 0.3,
            'rnn_units': 190,
            'fc_num_hidden_layers': 3,
            'fc_num_units': 280,
            'fc_drop': 0.5,
            'fc_activation': 'elu',
            'batch_size': 110,
            'epochs': 100,
            'optimizer': keras.optimizers.Adamax,
            'last_activation': 'linear',
            'initializer': 'he_normal'
        })

    return config

def test_model(model, test_sequence, test_bio_features, test_labels):
    # make predictions
    predictions = model.predict([test_sequence, test_bio_features])

    # calculate spearman correlation
    rho, pval = spearmanr(test_labels, predictions)

    return rho, pval



def train_model(model, train_sequence, train_bio_features, train_labels, val_sequence, val_bio_features, val_labels, config):
    # add early stopping
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, verbose=1)

    # save the best model
    model_checkpoint = keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

    # train the model
    history = model.fit([train_sequence, train_bio_features], train_labels,
                        batch_size=config['batch_size'], epochs=config['epochs'],
                        validation_data=([val_sequence, val_bio_features], val_labels),
                        callbacks=[early_stopping, model_checkpoint])
    
    return model, history
def get_model(config):
    seq_input_shape = SEQ_INPUT_SHAPE
    sequence_input = keras.Input(shape=seq_input_shape, dtype='float32', name='sequence_input')

    # Embedding layer
    embedded = layers.Embedding(5, config['em_dim'], input_length=seq_input_shape)(sequence_input)
    embedded = layers.SpatialDropout1D(config['em_drop'])(embedded)
    
    
    # RNN
    lstm = layers.LSTM(config['rnn_units'], dropout=config['rnn_drop'],
                        kernel_regularizer='l2', recurrent_regularizer='l2',
                        recurrent_dropout=config['rnn_rec_drop'], return_sequences=True, kernel_initializer=config['initializer'])  
    x = layers.Bidirectional(lstm)(embedded)
    x = layers.Flatten()(x)

    # Biological features
    bio_input_shape = BIO_INPUT_SHAPE
    biological_input = keras.Input(shape=bio_input_shape, dtype='float32', name='biological_input')
    x = layers.concatenate([x, biological_input])

    # Fully connected layers
    for _ in range(config['fc_num_hidden_layers']):
        x = layers.Dense(config['fc_num_units'], activation=config['fc_activation'],
                         kernel_initializer=config['initializer'])(x)
        x = layers.Dropout(config['fc_drop'])(x)

    # Output layer
    output = layers.Dense(1, activation=config['last_activation'],kernel_initializer=config['initializer'])(x)

    # Model
    model = keras.Model(inputs=[sequence_input, biological_input], outputs=output)
    model.compile(loss=losses.mean_squared_error, optimizer=config['optimizer'](learning_rate=0.002), metrics=['mse'])
    model.summary()

   
   

    return model



def main():
    config = get_config_for_enzyme(ENZYME)
    model = get_model(config)
    train_sequence, train_bio_features, train_labels = get_sequence_features_and_labels('train.csv', config)
    val_sequence, val_bio_features, val_labels = get_sequence_features_and_labels('valid.csv', config)
    test_sequence, test_bio_features, test_labels = get_sequence_features_and_labels('test.csv', config)

    model, history = train_model(model, train_sequence, train_bio_features, train_labels,
                                    val_sequence, val_bio_features, val_labels, config)
    

    
    rho, pval = test_model(model, test_sequence, test_bio_features, test_labels)
    print('rho: ', rho)
    

if __name__ == '__main__':
    main()
