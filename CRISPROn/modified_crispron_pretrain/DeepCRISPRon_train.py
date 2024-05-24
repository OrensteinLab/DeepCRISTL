#!/usr/bin/env python3
########################################################################
#
#  Copyright (c) 2021 by the contributors (see AUTHORS file)
#
#  This file is part of the CRISPRon
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  It is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this script, see file COPYING.
#  If not, see <http://www.gnu.org/licenses/>.
##########################################################################
import sys
import os
import shutil
import traceback
from collections import OrderedDict 
import numpy as np
from sklearn.model_selection import KFold
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.set_printoptions(threshold=np.inf)
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, callbacks, Model, optimizers, Input, utils
from tensorflow.keras.layers import Conv1D, Dropout, AveragePooling1D, Flatten, Dense, concatenate, SpatialDropout1D
from scipy import stats
from random import randint
import sys

from sklearn.model_selection import train_test_split


LEARN = 0.0001 
EPOCHS = 5000 
SEQ_C = "1"
VAL_C = "3"
VAL_G = "2"
BATCH_SIZE=500 
SEED = 0 
TEST_N = 0
TRAIN_PATH = 'co_pretrain.csv'

DO_SIZE_TESTS = True
TEST_PRETRAIN = True
DO_ALL = True
sizes = [5000, 10000, 15000, 20000]



print(
'LEARN=%f' % LEARN,
'EPOCHS=%i' % EPOCHS,
'SEQ_C=%s' % SEQ_C,
'VAL_C=%s' % VAL_C,
'VAL_G=%s' % VAL_G,
'BATCH_SIZE=%i' % BATCH_SIZE,
'SEED=%i' % SEED,
)

#length of input seq
eLENGTH=30
#depth of onehot encoding
eDEPTH=4


def read_seq_files(seq_c0, val_c0, val_g0):
    d = OrderedDict()
    fn = TRAIN_PATH
    with open(fn, 'rt') as f:
        head = f.readline().rstrip().split(',')
        if seq_c0.isnumeric():
            seq_c = int(seq_c0) -1
        else:
            if seq_c0 in head:
                seq_c = head.index(seq_c0)
            else:
                raise Exception

        if val_c0.isnumeric():
            val_c = int(val_c0) -1
        else:
            if val_c0 in head:
                val_c = head.index(val_c0)
            else:
                raise Exception

        if val_g0.isnumeric():
            val_g = int(val_g0) -1
        else:
            if val_g0 in head:
                val_g = head.index(val_g0)
            else:
                raise Exception
        f.seek(0)

        for i,l in enumerate(f):
            v = l.rstrip().split(',')
            s = v[seq_c]
            if not len(s) == eLENGTH:
                print('"%s" is not a string of length %d in line %d' % (s, eLENGTH, i+1))
                continue

            if s in d:
                print('"%s" is not unique in line %d', (s, i+1))
                continue

            try:
                e = float(v[val_c])
            except:
                print('no float value for "%s" in line %d', (s, i+1))
                e = 0.0
                continue
            try:
                g = float(v[val_g])
            except:
                print('no float value for "%s" in line %d', (s, i+1))
                g = 0.0
                continue

            d[s] = [g, e]
    return d

def onehot(x):
    z = list()
    for y in list(x):
        if y in "Aa":  z.append(0)
        elif y in "Cc": z.append(1)
        elif y in "Gg": z.append(2)
        elif y in "TtUu": z.append(3)
        else:
            print("Non-ATGCU character " + data[l])
            raise Exception
    return z

def set_data(DX, s):
    for j,x in enumerate(onehot(s)):
        DX[j][x] = 1


def preprocess_seq(data):
    
    dkeys = data.keys()
    DATA_X = np.zeros((len(dkeys),eLENGTH,eDEPTH), dtype=np.float32) # onehot
    DATA_G = np.zeros((len(dkeys)), dtype=np.float32) #deltaGb

    DATA_Y = np.zeros((len(dkeys)), dtype=np.float32) # efficiency

    for l, s in enumerate(dkeys):
        d = data[s]
        set_data(DATA_X[l], s)
        DATA_G[l] = d[0]
        DATA_Y[l] = d[1]

    return (list(dkeys), DATA_X, DATA_G, DATA_Y)



# check of inputs and setting a few base parameters
if SEED == 0:
    SEED = randint(0, sys.maxsize)
print('seed:', SEED)
tf.random.set_seed(SEED)
optimizer = optimizers.Adam(LEARN)

def get_model():

    #inputs
    #one hot
    inputs = list()
    input_c = Input(shape=(eLENGTH, eDEPTH,), name="input_onehot")
    inputs.append(input_c)

    #delta Gb
    input_g = Input(shape=(1,), name="input_dGB")
    inputs.append(input_g)

    #
    for_dense = list()

    #first convolution layer
    conv1_out = Conv1D(100, 3, activation='relu', input_shape=(eLENGTH,4,), name="conv_3")(input_c)
    conv1_dropout_out = Dropout(0.3, name="drop_3")(conv1_out)
    conv1_pool_out = AveragePooling1D(2, padding='SAME', name="pool_3")(conv1_dropout_out)
    conv1_flatten_out = Flatten(name="flatten_3")(conv1_pool_out)
    for_dense.append(conv1_flatten_out)

    #second convolution layer
    conv2_out = Conv1D(70, 5, activation='relu', input_shape=(eLENGTH,4,), name="conv_5")(input_c)
    conv2_dropout_out = Dropout(0.3, name="drop_5")(conv2_out)
    conv2_pool_out = AveragePooling1D(2, padding='SAME', name="pool_5")(conv2_dropout_out)
    conv2_flatten_out = Flatten(name="flatten_5")(conv2_pool_out)
    for_dense.append(conv2_flatten_out)

    #third convolution layer
    conv3_out = Conv1D(40, 7, activation='relu', input_shape=(eLENGTH,4,), name="conv_7")(input_c)
    conv3_dropout_out = Dropout(0.3, name="drop_7")(conv3_out)
    conv3_pool_out = AveragePooling1D(2, padding='SAME', name="pool_7")(conv3_dropout_out)
    conv3_flatten_out = Flatten(name="flatten_7")(conv3_pool_out)
    for_dense.append(conv3_flatten_out)

    #concatenation of conv layers and deltaGb layer
    if len(for_dense) == 1:
        concat_out = for_dense[0]
    else:
        concat_out = concatenate(for_dense)

    for_dense1 = list()

    #first dense (fully connected) layer
    dense0_out = Dense(80, activation='relu', name="dense_0")(concat_out)
    dense0_dropout_out = Dropout(0.3, name="drop_d0")(dense0_out)
    for_dense1.append(dense0_dropout_out)


    #Gb input used raw
    for_dense1.append(input_g)


    if len(for_dense1) == 1:
        concat1_out = for_dense1[0]
    else:
        concat1_out = concatenate(for_dense1)


    #first dense (fully connected) layer
    dense1_out = Dense(80, activation='relu', name="dense_1")(concat1_out)
    dense1_dropout_out = Dropout(0.3, name="drop_d1")(dense1_out)

    #second dense (fully connected) layer
    dense2_out = Dense(60, activation='relu', name="dense_2")(dense1_dropout_out)
    dense2_dropout_out = Dropout(0.3, name="drop_d2")(dense2_out)

    #output layer
    output = Dense(1, name="output")(dense2_dropout_out)

    #model construction
    model= Model(inputs=inputs, outputs=[output])
    model.summary()
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    return model
#utils.plot_model(model, to_file=str(TEST_N) + '.model.png', show_shapes=True, dpi=600)




tinput = list()
vinput = list()

print('reading training data')
d = read_seq_files(SEQ_C, VAL_C, VAL_G)
(s, x, g, y) = preprocess_seq(d)


#
# Get models for each size
#

for size in sizes:
    print("SEED", SEED)
    # Number of folds
    n_splits = 6
    kf = KFold(n_splits=n_splits)

    # Convert your data to numpy arrays if they aren't already, to ensure compatibility with sklearn's KFold.
    X = np.array(x)
    G = np.array(g)
    Y = np.array(y)

    # sample the data after shuffling
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    G = G[indices]
    Y = Y[indices]

    # sample the data to the size
    X = X[:size]
    G = G[:size]
    Y = Y[:size]

    split_num = 1
    for train_index, val_index in kf.split(X):
        # Splitting the data for this fold
        x_train, x_val = X[train_index], X[val_index]
        g_train, g_val = G[train_index], G[val_index]
        y_train, y_val = Y[train_index], Y[val_index]

        tinput = []
        vinput = []

        tinput.append(x_train)
        vinput.append(x_val)

        tinput.append(g_train)
        vinput.append(g_val)

        tinput = tuple(tinput)
        vinput = tuple(vinput)

        es = callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100) # Patience was 150
        mc = callbacks.ModelCheckpoint(str(split_num) + f'_{size}.model.best', verbose=1, save_best_only=True)
        model = get_model()
        # Fit the model for this fold
        history = model.fit(tinput, y_train, validation_data=(vinput, y_val), batch_size=BATCH_SIZE, \
                            epochs=EPOCHS, use_multiprocessing=True, workers=16, verbose=2, callbacks=[es, mc])
        
        print(history.history)
        for key in history.history.keys():
            # print key and value
            print(key, history.history[key])

        split_num += 1

#
# Test pretrain on each size
#


# split X G and Y to train test
X_train, X_test, G_train, G_test, Y_train, Y_test = train_test_split(x, g, y, test_size=0.2, random_state=42)
for size in sizes:

    print("SEED", SEED)
    # Number of folds
    n_splits = 6
    kf = KFold(n_splits=n_splits)

    # Convert your data to numpy arrays if they aren't already, to ensure compatibility with sklearn's KFold.
    X = np.array(X_train)
    G = np.array(G_train)
    Y = np.array(Y_train)

    # sample the data after shuffling
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    G = G[indices]
    Y = Y[indices]

    # sample the data to the size
    X = X[:size]
    G = G[:size]
    Y = Y[:size]

    ensemble = []
    for train_index, val_index in kf.split(X):
        # Splitting the data for this fold
        x_train, x_val = X[train_index], X[val_index]
        g_train, g_val = G[train_index], G[val_index]
        y_train, y_val = Y[train_index], Y[val_index]

        tinput = []
        vinput = []

        tinput.append(x_train)
        vinput.append(x_val)

        tinput.append(g_train)
        vinput.append(g_val)

        tinput = tuple(tinput)
        vinput = tuple(vinput)

        es = callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100) # Patience was 150
        model = get_model()
        # Fit the model for this fold
        history = model.fit(tinput, y_train, validation_data=(vinput, y_val), batch_size=BATCH_SIZE, \
                            epochs=EPOCHS, use_multiprocessing=True, workers=16, verbose=2, callbacks=[es])
        ensemble.append(model)

    # predict on test
    X_TEST = np.array(X_test)
    G_TEST = np.array(G_test)
    Y_TEST = np.array(Y_test)

    predictions = []
    for model in ensemble:
        predictions.append(model.predict([X_TEST, G_TEST]))
    
    predictions = np.array(predictions)
    predictions = np.mean(predictions, axis=0)
    # get spearman with test
    spearman = stats.spearmanr(predictions, Y_TEST)
    print(f"Size Test: {size}, Spearman: {spearman}")
        

# Get normal models
split_num = 1
for _ in range (5): #TODO: remove when doint last one
    print("SEED", SEED)
    # Number of folds
    n_splits = 6
    kf = KFold(n_splits=n_splits)

    # Convert your data to numpy arrays if they aren't already, to ensure compatibility with sklearn's KFold.
    X = np.array(x)
    G = np.array(g)
    Y = np.array(y)


    for train_index, val_index in kf.split(X):
        # Splitting the data for this fold
        x_train, x_val = X[train_index], X[val_index]
        g_train, g_val = G[train_index], G[val_index]
        y_train, y_val = Y[train_index], Y[val_index]

        tinput = []
        vinput = []

        tinput.append(x_train)
        vinput.append(x_val)

        tinput.append(g_train)
        vinput.append(g_val)

        tinput = tuple(tinput)
        vinput = tuple(vinput)

        es = callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100) # Patience was 150
        mc = callbacks.ModelCheckpoint(str(split_num) + '.model.best', verbose=1, save_best_only=True)
        model = get_model()
        # Fit the model for this fold
        history = model.fit(tinput, y_train, validation_data=(vinput, y_val), batch_size=BATCH_SIZE, \
                            epochs=EPOCHS, use_multiprocessing=True, workers=16, verbose=2, callbacks=[es, mc])
        
        print(history.history)
        for key in history.history.keys():
            # print key and value
            print(key, history.history[key])

        split_num += 1