import numpy as np
import os
import pickle

# My Scripts
from scripts import models_util
from scripts import training_util
from scripts import testing_util


def concantenate_test_data(config, X, X_biofeat, y, DataHandler):
    if config.enzyme == 'multi_task':
        a=0 # TODO
    else:
        X = np.concatenate((X, DataHandler['test'].enzymes_seq[config.enzyme].X))
        X_biofeat = np.concatenate((X_biofeat, DataHandler['test'].enzymes_seq[config.enzyme].X_biofeat))
        y = np.concatenate((y, DataHandler['test'].enzymes_seq[config.enzyme].y))
    return X, X_biofeat, y

def create_data(config, DataHandler):
    # Create main dir
    data_dir = f'data/pre_train/{config.pre_train_data}/'
    if config.simulation_type == 'cross_v':
        data_dir += f'{config.enzyme}_10_fold/'
    elif config.simulation_type == 'full_cross_v':
        data_dir += f'full_{config.enzyme}_10_fold/'

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if os.path.exists(data_dir + '9_fold/'):
        print('Data allready created')
        return

    if (config.pre_train_data == 'DeepHF_full') and not (config.transfer_learning):
        weighted_loss = True
    else:
        weighted_loss = False

    X = np.concatenate((DataHandler['X_train'], DataHandler['X_valid']))
    X_biofeat = np.concatenate((DataHandler['X_biofeat_train'], DataHandler['X_biofeat_valid']))
    y = np.concatenate((DataHandler['y_train'], DataHandler['y_valid']))

    # If full cross validation, add test data -> Note: this is already done on the file level so this is redundant
    # if config.simulation_type == 'full_cross_v':
    #     X, X_biofeat, y = concantenate_test_data(config, X, X_biofeat, y, DataHandler)

    if weighted_loss:
        confidence = np.concatenate((DataHandler['conf_train'], DataHandler['conf_valid']))
        if config.simulation_type == 'full_cross_v':
            if config.enzyme == 'multi_task':
                a=0 # TODO
            else:
                confidence = np.concatenate((confidence, DataHandler['test'].enzymes_seq[config.enzyme].confidence))

    perm = np.random.permutation(X.shape[0])
    val_size = int(X.shape[0] / 10)

    for ind in range(10):
        # Create fold dir
        fold_dir = data_dir + f'{ind}_fold/'
        os.mkdir(fold_dir)

        # Split val train
        valid_ind = perm[ind*val_size:(ind+1)*val_size]

        X_valid = X[valid_ind]
        X_train = np.delete(X, valid_ind, axis=0)

        X_biofeat_valid = X_biofeat[valid_ind]
        X_biofeat_train = np.delete(X_biofeat, valid_ind, axis=0)

        y_valid = y[valid_ind]
        y_train = np.delete(y, valid_ind, axis=0)

        if weighted_loss:
            conf_valid = confidence[valid_ind]
            conf_train = np.delete(confidence, valid_ind, axis=0)

            pickle.dump(conf_valid, open(fold_dir + f'conf_valid.pkl', "wb"))
            pickle.dump(conf_train, open(fold_dir + f'conf_train.pkl', "wb"))

        pickle.dump(X_valid, open(fold_dir + f'X_valid.pkl', "wb"))
        pickle.dump(X_train, open(fold_dir + f'X_train.pkl', "wb"))
        pickle.dump(X_biofeat_valid, open(fold_dir + f'X_biofeat_valid.pkl', "wb"))
        pickle.dump(X_biofeat_train, open(fold_dir + f'X_biofeat_train.pkl', "wb"))
        pickle.dump(y_valid, open(fold_dir + f'y_valid.pkl', "wb"))
        pickle.dump(y_train, open(fold_dir + f'y_train.pkl', "wb"))




def load_fold_data(config, DataHandler, k):
    data_dir = f'data/pre_train/{config.pre_train_data}/'
    if config.simulation_type == 'cross_v':
        data_dir += f'{config.enzyme}_10_fold/'
    elif config.simulation_type == 'full_cross_v':
        data_dir += f'full_{config.enzyme}_10_fold/'

    data_dir += f'{k}_fold/'

    DataHandler['X_valid'] = pickle.load(open(data_dir + 'X_valid.pkl', "rb"))
    DataHandler['X_train'] = pickle.load(open(data_dir + 'X_train.pkl', "rb"))
    DataHandler['X_biofeat_valid'] = pickle.load(open(data_dir + 'X_biofeat_valid.pkl', "rb"))
    DataHandler['X_biofeat_train'] = pickle.load(open(data_dir + 'X_biofeat_train.pkl', "rb"))
    DataHandler['y_valid'] = pickle.load(open(data_dir + 'y_valid.pkl', "rb"))
    DataHandler['y_train'] = pickle.load(open(data_dir + 'y_train.pkl', "rb"))

    if (config.pre_train_data == 'DeepHF_full') and not config.transfer_learning:
        DataHandler['conf_valid'] = pickle.load(open(data_dir + 'conf_valid.pkl', "rb"))
        DataHandler['conf_train'] = pickle.load(open(data_dir + 'conf_train.pkl', "rb"))


    return DataHandler


def train_10_fold(config, DataHandler):
    create_data(config, DataHandler)
    spearman_result = []

    for k in range(10):
        DataHandler = load_fold_data(config, DataHandler, k)
        model, callback_list = models_util.get_model(config, DataHandler)
        history = training_util.train_model(config, DataHandler, model, callback_list)
        if config.simulation_type == 'cross_v':
            spearman_result.append(testing_util.test_model(config, model, DataHandler['test']))


    if config.simulation_type == 'cross_v':
        results_path = 'results/transfer_learning/' if config.transfer_learning else 'results/pre_train/'
        if not os.path.exists('results'):
            os.mkdir('results')
        if not os.path.exists(results_path):
            os.mkdir(results_path)
        results_path += f'{config.pre_train_data}_{config.enzyme}_cross_v.txt'
        f = open(results_path, 'w')

        res = dict()
        for dic in spearman_result:
            for list in dic:
                if list in res:
                    res[list].append(dic[list])
                else:
                    res[list] = [dic[list]]


        for enzyme, spearmans in res.items():
            mean = sum(spearmans) / len(spearmans)
            f.write(f'{enzyme}: spearmans - {spearmans}, mean - {mean}\n')

        f.close()
