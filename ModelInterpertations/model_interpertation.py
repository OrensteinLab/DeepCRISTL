import os
import numpy as np
import pandas as pd
import logomaker
import matplotlib.pyplot as plt
import math

from keras.models import load_model, Input, Model
from keras.layers import SpatialDropout1D, LSTM, Bidirectional, Flatten, concatenate, Dense, Dropout
import tensorflow as tf

from see_rnn import get_gradients, visuals_rnn, inspect_rnn, inspect_gen

# my imports
from scripts import configurations as cfg
from scripts_tl import configurations_tl as cfg_tl
from scripts import data_handler as dh
from scripts_tl import data_handler_tl as dh_tl


def get_model(config, path):
    # Loading model
    old_model = load_model(path)

    # Replacing model with a new model with same weights for interpretability
    sequence_input = Input(name='seq_input', shape=(22,5))
    identity = np.identity(5)
    id = tf.matmul(sequence_input, identity) # This layer is used for technical reasons, so we can calculate the gradients of the inputs

    # Embedding layer conversion
    emb_mat = old_model.get_weights()[0]
    emb_mat = tf.convert_to_tensor(emb_mat)
    embedded = tf.matmul(id, emb_mat)
    embedded = SpatialDropout1D(0.99)(embedded)
    x = embedded

    # RNN
    lstm = LSTM(config.rnn_units,  kernel_regularizer='l2', recurrent_regularizer='l2', return_sequences=True)
    x = Bidirectional(lstm)(x)
    x = Flatten()(x)

    # Biological features
    biological_input = Input(name = 'bio_input', shape = (14,))
    x = concatenate([x, biological_input])

    # Fully connected layers
    for l in range(config.fc_num_hidden_layers):
        x = Dense(config.fc_num_units, activation=config.fc_activation)(x)
        x = Dropout(0.99)(x)



    # Output layer
    output = Dense(1, activation=config.last_activation, name='output')(x)

    # Defining model
    model = Model(inputs=[sequence_input, biological_input], outputs=[output])
    model.compile(loss='mse', optimizer=config.optimizer(lr=0.002))
    # model.summary()
    #
    # old_model.summary()
    old_weights = old_model.get_weights()[1:]
    model.set_weights(old_weights)


    return model

def save_logos_diff(logo_dict):
    a=0


def create_and_save_logo(PWM_df, results_path):
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Create logo for each iteration
    # fig, axs = plt.subplots(5, 2)

    # for init_mat, logo_df in logo_dict.items():
    IG_logo = logomaker.Logo(PWM_df,
                             shade_below=.5,
                             fade_below=.5,
                             color_scheme='classic',
                             # ax=axs[init_mat%5,math.floor(init_mat/5)]
                             )
    IG_logo.ax.set_xticks(range(len(PWM_df)))
    size =14
    IG_logo.ax.set_xticklabels(np.arange(1, 22), fontsize=size)
    # IG_logo.ax.set_yticklabels(np.array((0.0, 0.2, 0.4, 0.6, 0.8, 1.0)), fontsize=12)
    IG_logo.ax.tick_params(axis='y', labelsize=size)
    IG_logo.ax.set_xlabel('Nucleotide position', fontsize=size)
    IG_logo.ax.set_ylabel('Importance score', fontsize=size)

    a=0
    plt.show()
    # path = result_path + f'logo.png'
    # plt.savefig(path)

    # fig = plt.figure()

    # dfs = logo_dict.values()
    # averages = pd.concat([each.stack() for each in dfs], axis=1).apply(lambda x: x.mean(), axis=1).unstack()
    # IG_logo = logomaker.Logo(averages,
    #                          shade_below=.5,
    #                          fade_below=.5,
    #                          color_scheme='classic',
    #                          )
    # path = result_path + f'avg_grna.png'
    # plt.savefig(path)


expirements = ['xu2015TrainHl60', 'chari2015Train293T', 'hart2016-Rpe1Avg', 'hart2016-Hct1162lib1Avg',
               'hart2016-HelaLib1Avg', 'hart2016-HelaLib2Avg','xu2015TrainKbm7', 'doench2014-Hs' , 'doench2014-Mm',
               'doench2016_hg19'] #'leenay'

expirements = ['hart2016-Rpe1Avg']
data_types = {'DeepHF':[['multi_task'], 'pre_train/DeepHF_old'], 'U6T7': [expirements, 'transfer_learning']}
# data_types = {'leenay_anat': [expirements, 'transfer_learning']}
data_types = {'U6T7': [expirements, 'transfer_learning']}

logo_dict = {}

for data_category, (data_type_arr, pre_path) in data_types.items():
    for data_type in data_type_arr:
        result_path = f'results/{data_category}/{data_type}/'
        # if os.path.exists(result_path + f'logo.png'):
        #     continue

        print(f'Starting logo on {data_type}')

        if data_category == 'DeepHF':
            config = cfg.get_parser(data_type)
            model_path = f'../DeepCRISTL2/models/{pre_path}/{data_type}/model_0/model'

        else:
            config = cfg_tl.get_parser()
            model_path = f'../DeepCRISTL2/models/{pre_path}/{data_type}/set0/DeepHF_old/multi_task/gl_tl/model_0/model'
            a=0

        # Loading & preparing model
        model = get_model(config, model_path)

        # Loading Data & calculating average bio features input
        if data_category == 'DeepHF':
            if data_type == 'multi_task':
                DataHandler = dh.get_data(config)['test']
                wt_biofeat = DataHandler.enzymes_seq['wt'].X_biofeat
                esp_biofeat = DataHandler.enzymes_seq['esp'].X_biofeat
                hf_biofeat = DataHandler.enzymes_seq['hf'].X_biofeat

                X_biofeat = np.concatenate((wt_biofeat, esp_biofeat, hf_biofeat), axis=0)
                mean_biofeat = np.mean(X_biofeat, axis = 0)
                mean_biofeat = np.expand_dims(mean_biofeat, axis=0)
                mean_biofeat = tf.convert_to_tensor(mean_biofeat)

            else:
                DataHandler = dh.get_data(config)['test'].enzymes_seq[data_type]
                X_biofeat = DataHandler.X_biofeat
                mean_biofeat = np.mean(X_biofeat, axis = 0)
                mean_biofeat = np.expand_dims(mean_biofeat, axis=0)
                mean_biofeat = tf.convert_to_tensor(mean_biofeat)

        else:
            config.tl_data_category = data_category
            config.tl_data = data_type
            DataHandler = dh_tl.get_data(config, 0)

            X_biofeat = np.concatenate((DataHandler['X_biofeat_train'], DataHandler['X_biofeat_valid'], DataHandler['X_biofeat_test']), axis=0)
            mean_biofeat = np.mean(X_biofeat, axis = 0)
            mean_biofeat = np.expand_dims(mean_biofeat, axis=0)
            mean_biofeat = tf.convert_to_tensor(mean_biofeat)

            a=0

        # Preparing zeros & ones matrix
        mat = np.ones(shape=(22, 5))
        mat[0, :] = 0
        mat[:, 0] = 0
        start_val = np.zeros(shape=(22, 5))
        start_val[0, 0] = 1.
        y_opt = np.array([100.])

        # for j in range(10):
        # print(f'Starting simulation {j}')
        x = tf.ones(shape=(1, 22, 5)) * 0.25 # Initial input matrix
        lr = 0.01
        # Calculating optimal input
        for i in range(1000):
            x = x * mat + start_val
            input = [x, mean_biofeat]
            grads = get_gradients(model, 1, input, y_opt)
            x = x - grads * lr

            if i % 50 == 0:
                pred = model(input)
                print(i, pred)

                # if i% 100 == 0:
                #     lr = lr/2

        # Receiving logo
        x = tf.squeeze(x)
        x = x.numpy()
        x_new = x[1:, 1:]
        PWM_df = pd.DataFrame(x_new, columns=['A', 'T', 'C', 'G'])
        logo_dict[data_type] = PWM_df

        create_and_save_logo(PWM_df, result_path)

save_logos_diff(logo_dict)
a=0
#
# import imageio
# images_arr = []
# for data_category, (data_type_arr, pre_path) in data_types.items():
#     for data_type in data_type_arr:
#         im = imageio.imread(f'results/{data_category}/{data_type}/logo.png')
#         # im = im[25:225]
#         np_frame = np.array(im)
#         np_frame = np_frame[25:225, 80:980, :]
#         images_arr.append([np_frame, data_type])
#
#
# #
# #
# # import matplotlib.pyplot as plt
# # import numpy as np
#
# # settings
# figsize = [5, 4]     # figure size, inches
#
# import math
# fig, ax = plt.subplots(nrows=math.ceil(len(images_arr)/2), ncols=2, figsize=figsize)
#
# for i, axi in enumerate(ax.flat):
#     axi.axis("off")
#     if i > len(images_arr) - 1:
#         break
#     axi.imshow(images_arr[i][0])
#     axi.set_title(images_arr[i][1], fontsize=3)
#
# plt.tight_layout(True)
# plt.show()
#
# # fig, axs = plt.subplots(5, 2)
