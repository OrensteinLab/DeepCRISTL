from keras.models import load_model, Model
from keras.layers import Input, Dense
from keras.layers.merge import concatenate
import os
import numpy as np
import scipy as sp

from scripts import models_util


def load_all_models(config):
    models_dir = 'models/transfer_learning/' if config.transfer_learning else 'models/pre_train/'
    models_dir += f'{config.pre_train_data}/{config.enzyme}/'

    all_models = []
    model_ind = 0
    model_path = models_dir + 'model_0/model'
    while os.path.exists(model_path):
        print(f'Loading model_{model_ind}')
        all_models.append(load_model(model_path))
        model_ind += 1
        model_path = models_dir + f'model_{model_ind}/model'

    return all_models


# def define_stacked_model(config, members):
#     # update all layers in all models to not be trainable
#     for i in range(len(members)):
#         model = members[i]
#         for layer in model.layers:
#             # make not trainable
#             layer.trainable = False
#             # rename to avoid 'unique layer name' issue
#             layer.name = 'ensemble_' + str(i+1) + '_' + layer.name
#
#
#     # define multi head input
#     ensemble_visible = []
#     for model in members:
#         ensemble_visible += model.input
#
#     a=0
#     # concatenate merge output from each model
#     ensemble_outputs = [model.output for model in members]
#     merge = concatenate(ensemble_outputs)
#     # hidden = Dense(2, activation='relu')(merge)
#     output = Dense(1, activation='sigmoid')(merge)
#     model = Model(inputs=ensemble_visible, outputs=output)
#     # plot graph of ensemble
# 	# plot_model(model, show_shapes=True, to_file='model_graph.png')
# 	# compile
#     model.compile(loss='mse', optimizer=config.optimizer(lr=0.002))
#     return model

# def fit_stacked_model(config, model, DataHandler):
#     # prepare input data
#     train_input = []
#     valid_input = []
#
#     for i in range(int(len(model.input)/2)):
#         train_input += [DataHandler['X_train'], DataHandler['X_biofeat_train']]
#         valid_input += [DataHandler['X_valid'], DataHandler['X_biofeat_valid']]
#
#     y_train = DataHandler['y_train']
#     y_val = DataHandler['y_valid']
#
#     loss_weights = DataHandler['conf_train']
#     valid_loss_weights = DataHandler['conf_valid']
#     weight_scale = np.mean(loss_weights)
#     loss_weights = loss_weights / weight_scale
#     valid_loss_weights = valid_loss_weights / weight_scale
#
#     _, callback_list = models_util.get_model(config, DataHandler)
#     # fit model
#     history = model.fit(train_input,
#                         y_train,
#                         batch_size=config.batch_size,
#                         epochs=20, #config.epochs,
#                         verbose=2,
#                         validation_data=(valid_input, y_val, valid_loss_weights),
#                         shuffle=True,
#                         callbacks=callback_list,
#                         sample_weight=loss_weights
#                         )


# def test_ensemble(config, model, DataHandler):
#     test_input = []
#     if config.enzyme == 'multi_task':
#         a=0#TODO
#     else:
#         for i in range(int(len(model.input) / 2)):
#             test_input += [DataHandler['test'].enzymes_seq[config.enzyme].X,
#                            DataHandler['test'].enzymes_seq[config.enzyme].X_biofeat]
#         test_true_label = DataHandler['test'].enzymes_seq[config.enzyme].y
#
#         test_prediction = model.predict(test_input)
#         spearman = sp.stats.spearmanr(test_true_label, test_prediction)[0]
#         print(f'Enzyme: {config.enzyme}, Spearman: {spearman}')
#     a=0



def test_means(config, all_models, DataHandler):
    print('\nTesting models:')
    if config.enzyme == 'multi_task':
        enzymes = ['wt', 'esp', 'hf']
        spearman_result = {}
        for enzyme in enzymes:
            test_input = [DataHandler['test'].enzymes_seq[enzyme].X, DataHandler['test'].enzymes_seq[enzyme].X_biofeat]
            test_true_label = DataHandler['test'].enzymes_seq[enzyme].y
            predictions = []
            for ind, model in enumerate(all_models):
                print(f'Testing model_{ind} with {enzyme}')
                test_prediction = model.predict(test_input)
                predictions.append(test_prediction)

            finall_pred = np.zeros((test_prediction.shape[0], 1))
            for pred in predictions:
                finall_pred += pred
            finall_pred /= len(predictions)
            spearmanr = sp.stats.spearmanr(test_true_label, finall_pred)
            spearman_result[enzyme] = spearmanr
        for enzyme in enzymes:
            print(f'Enzyme: {enzyme}, Spearman: {spearman_result[enzyme]}')
    else:
        predictions = []
        test_input = [DataHandler['test'].enzymes_seq[config.enzyme].X, DataHandler['test'].enzymes_seq[config.enzyme].X_biofeat]
        test_true_label = DataHandler['test'].enzymes_seq[config.enzyme].y
        for ind, model in enumerate(all_models):
            print(f'Testing model_{ind}')
            test_prediction = model.predict(test_input)
            predictions.append(test_prediction)

        finall_pred = np.zeros((test_prediction.shape[0], 1))
        for pred in predictions:
            finall_pred += pred
        finall_pred /= len(predictions)
        spearmanr = sp.stats.spearmanr(test_true_label, finall_pred)
        print(f'Data: {config.pre_train_data}, Enzyme: {config.enzyme}, Spearman: {spearmanr}')


def train_ensemble(config, DataHandler):
    config.save_model = False
    all_models = load_all_models(config)
    test_means(config, all_models, DataHandler)
    # stacked_model = define_stacked_model(config, all_models)
    # fit_stacked_model(config, stacked_model, DataHandler)
    # test_ensemble(config, stacked_model, DataHandler)

    a=0