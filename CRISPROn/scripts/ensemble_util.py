import os
from keras.models import load_model
import numpy as np
import scipy as sp


def load_all_models(config, model_string=''):
    models_dir = f'tl_models/transfer_learning/{config.tl_data}/set{config.set}/{config.train_type}/'

    all_models = []
    model_ind = 0
    model_path = models_dir + 'model_0/model'
    while os.path.exists(model_path):
        print(f'Loading model_{model_ind}')
        all_models.append(load_model(model_path))
        model_ind += 1
        model_path = models_dir + f'model_{model_ind}/model'

    return all_models

def test_means(config, all_models, DataHandler):
    print('\nTesting models:')

    predictions = []
    test_input = [DataHandler['X_test'], DataHandler['dg_test']]
    test_true_label = DataHandler['y_test']
    for ind, model in enumerate(all_models):
        print(f'Testing model_{ind}')
        test_prediction = model.predict(test_input)
        predictions.append(test_prediction)

    finall_pred = np.zeros((test_prediction.shape[0], 1))
    for pred in predictions:
        finall_pred += pred
    finall_pred /= len(predictions)
    spearmanr = sp.stats.spearmanr(test_true_label, finall_pred)
    print(f'Spearman: {spearmanr}')
    return spearmanr

def train_ensemble(config, DataHandler, model_string=''):
    config.save_model = False
    all_models = load_all_models(config, model_string=model_string)
    spearmanr = test_means(config, all_models, DataHandler)
    return spearmanr

def test_ensemble(config, DataHandler, models):
    spearmanr = test_means(config, models, DataHandler)
    return spearmanr