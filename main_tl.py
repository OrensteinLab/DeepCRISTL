import tensorflow as tf
import numpy as np
import keras
from keras.optimizers import *

# my scripts

# General scripts
from scripts import models_util
import os

# Transfer learning scripts
from scripts_tl import configurations_tl as cfg_tl
from scripts_tl import preprocess_tl
from scripts_tl import data_handler_tl as dh_tl
from scripts_tl import training_util_tl
from scripts_tl import testing_util_tl
from scripts_tl import hyper_parameter_search_tl as hps_tl
from scripts_tl import cross_validation_tl as cv_tl
from scripts_tl import ensemble_util_tl
from scripts_tl import postprocess_tl


if __name__ == '__main__':
    # Using only needed memory on GPU
    # con = tf.compat.v1.ConfigProto()
    # con.gpu_options.allow_growth = True
    # tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=con))

    # Get configs
    config = cfg_tl.get_parser()

    if config.simulation_type == 'preprocess':
        preprocess_tl.prepare_inputs(config)

    if config.simulation_type == 'train':
        config.epochs = 500

        config.save_model = False
        # Initializing random seed
        np.random.seed(1234)

        print('simulation_type = train')

        print('Receiving DataHandler')
        DataHandler = dh_tl.get_data(config)
        print('Building model')
        model, callback_list = models_util.load_pre_train_model(config, DataHandler)
        # model.layers[1].trainable = False
        history = training_util_tl.train_model(config, DataHandler, model, callback_list)
        testing_util_tl.test_model(config, model, DataHandler)

    if config.simulation_type == 'param_search':
        config.save_model = False
        DataHandler = dh_tl.get_data(config)
        hps_tl.param_search(config, DataHandler)

    if config.simulation_type == 'train_full':
        config.save_model = False
        config.set = 0
        DataHandler = dh_tl.get_data(config, set=0)
        cv_tl.train_10(config, DataHandler)

    if config.simulation_type == 'cross_v_HPS':
        config.save_model = False
        DataHandler = dh_tl.get_data(config)
        cv_tl.cross_v_HPS(config, DataHandler)

    if config.simulation_type == 'ensemble':
        DataHandler = dh_tl.get_data(config)
        ensemble_util_tl.train_ensemble(config, DataHandler)

    if config.simulation_type == 'data_size_scan':
        config.save_model = False
        DataHandler = dh_tl.get_data(config)
        cv_tl.data_size_scan(config, DataHandler)


    if config.simulation_type == 'full_sim':
        if config.tl_data == 'ALL_ORIGINAL_DATA':
            datasets = ['leenay', 'chari2015Train293T','doench2014-Hs','doench2014-Mm','doench2016_hg19','hart2016-Hct1162lib1Avg','hart2016-HelaLib1Avg','hart2016-HelaLib2Avg','hart2016-Rpe1Avg','morenoMateos2015','xu2015TrainHl60','xu2015TrainKbm7']
        else:
            datasets = [config.tl_data]

        for dataset in datasets:
            config.tl_data = dataset
            print(f'Running full simulation for {config.tl_data} dataset')

            for set in range(5):
                print(f'Running on set {set}')
                config.set = set
                DataHandler = dh_tl.get_data(config, set)


                train_types = ['full_tl', 'LL_tl', 'gl_tl', 'no_em_tl', 'no_tl', 'no_pre_train']
                config.enzyme = 'multi_task'
                for train_type in train_types:
                    print(f'#################### Running {config.enzyme} with {train_type} model #############################')
                    config.train_type = train_type
                    config.epochs = 1000
                    config.save_model = False

                    print(f'Running cross_v_HPS for {config.enzyme} with {train_type} model')
                    if config.train_type == 'no_pre_train':
                        config.epochs = 200
                        config.batch_size = 190
                        config.init_lr = 0.027
                        config.optimizer = SGD
                        opt_epochs = cv_tl.cross_v_HPS(config, DataHandler)
                        config.epochs = opt_epochs
                    elif config.train_type == 'no_tl':
                        config.epochs = 0
                    else:
                        opt_epochs = cv_tl.cross_v_HPS(config, DataHandler)
                        config.epochs = opt_epochs

                    print(f'Running full_train for {config.enzyme} with {train_type} model')
                    config.save_model = True
                    mean = cv_tl.train_10(config, DataHandler)

                    print(f'Running ensemble for {config.enzyme} with {train_type} model')
                    spearmanr = ensemble_util_tl.train_ensemble(config, DataHandler)

                    testing_util_tl.save_results(config, config.enzyme, train_type, mean, spearmanr)
                    keras.backend.clear_session()

    if config.simulation_type == 'postprocess':
        postprocess_tl.postprocess(config)

    if config.simulation_type == 'get_learning_curve':
        postprocess_tl.get_learning_curve(config)


    if config.simulation_type == 'test_KBM7_model':
        postprocess_tl.test_KBM7_model(config)

