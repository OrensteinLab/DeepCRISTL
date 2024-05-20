from scripts import configurations as cfg
from scripts import preprocess
from scripts import data_handler as dh
from scripts import cross_validation_tl as cv_tl
from scripts import ensemble_util
from scripts import testing_util
from scripts import  hyper_parameter_search as hps
from scripts import postprocess
import keras
import time
import numpy as np
import os

#interperter = ModelInterpertation

if __name__ == '__main__':
    # Get configs
    config = cfg.get_parser()

    if config.simulation_type == 'preprocess':
        if config.tl_data == 'ALL_U6T7_DATA':
            datasets = ['chari2015Train293T','doench2014-Hs','doench2014-Mm','doench2016_hg19', 'doench2016plx_hg19','hart2016-Hct1162lib1Avg','hart2016-HelaLib1Avg','hart2016-HelaLib2Avg','hart2016-Rpe1Avg','morenoMateos2015','xu2015TrainHl60','xu2015TrainKbm7']
        elif config.tl_data == 'ALL_ORIGINAL_DATA':
            datasets = ['leenay', 'chari2015Train293T','doench2014-Hs','doench2014-Mm','doench2016_hg19', 'doench2016plx_hg19','hart2016-Hct1162lib1Avg','hart2016-HelaLib1Avg','hart2016-HelaLib2Avg','hart2016-Rpe1Avg','morenoMateos2015','xu2015TrainHl60','xu2015TrainKbm7']
        else:
            datasets = [config.tl_data]

        for dataset in datasets:
            config.tl_data = dataset
            print(f'Running preprocess for {config.tl_data} dataset')
            preprocess.prepare_inputs(config)

    if config.simulation_type == 'compare_sizes':
        config.tl_data = 'morenoMateos2015'
        sizes = ['no_tl', 5000, 10000, 15000, 20000,'full']
        for size in sizes:


            print(f'Running simulation for {config.tl_data} dataset on size {size}')
            if size == 'no_tl':
                train_types = ['no_tl']
                model_string = ''
            else:
                train_types = [ 'LL_tl', 'gl_tl']
                if size == 'full':
                    model_string = ''
                else:
                    model_string = f'_{size}'


            gl_spearmans = []
            no_tl_spearmans = []
            for set in range(5):
                print(f'Running on set {set}')
                config.set = set
                DataHandler = dh.get_data(config, set)
                


                for train_type in train_types:
                    # start counting time

                    print(f'#################### Running {train_type} model #############################')
                    config.train_type = train_type
                    config.save_model = False
                    print(f'Running cross_v_HPS with {train_type} model')
                    config.epochs = 100
                    opt_epochs = cv_tl.cross_v_HPS(config, DataHandler, model_string=model_string)
                    config.epochs = opt_epochs
    
                    print(f'Running full_train with {train_type} model')
                    config.save_model = True
                    mean, models = cv_tl.train_6(config, DataHandler, model_string=model_string, return_model=True)


                    print(f'Running ensemble with {train_type} model')
                    #spearmanr = ensemble_util.train_ensemble(config, DataHandler)
                    spearmanr = ensemble_util.test_ensemble(config, DataHandler, models)

                    if train_type == 'gl_tl':
                        gl_spearmans.append(spearmanr[0])
                    if train_type == 'no_tl':
                        no_tl_spearmans.append(spearmanr[0])

                    keras.backend.clear_session()
            if size == 'no_tl':
                print(f'Spearmans for no_tl on morenoMateos2015: {no_tl_spearmans}')
            else:
                print(f'Spearmans for gl_tl on morenoMateos2015 with size {size}: {gl_spearmans}')

            # delete the created models
            path = f'tl_models/transfer_learning/morenoMateos2015'
            for root, dirs, files in os.walk(path):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))




    if config.simulation_type == 'full_sim':
        if config.tl_data == 'ALL_ORIGINAL_DATA':
            datasets = ['leenay', 'chari2015Train293T','doench2014-Hs','doench2014-Mm','doench2016_hg19', 'doench2016plx_hg19','hart2016-Hct1162lib1Avg','hart2016-HelaLib1Avg','hart2016-HelaLib2Avg','hart2016-Rpe1Avg','morenoMateos2015','xu2015TrainHl60','xu2015TrainKbm7']
        else:
            datasets = [config.tl_data]

        for dataset in datasets:
            config.tl_data = dataset
            print(f'Running full simulation for {config.tl_data} dataset')

            train_types = ['full_tl', 'LL_tl', 'gl_tl', 'no_tl', 'no_pre_train', 'no_conv_tl']
            train_times = {'full_tl': [], 'LL_tl': [], 'gl_tl': [], 'no_tl': [], 'no_pre_train': [], 'no_conv_tl': []}
            epochs_chosen = {'full_tl': [], 'LL_tl': [], 'gl_tl': [], 'no_tl': [], 'no_pre_train': [], 'no_conv_tl': []}
            for set in range(5):
                print(f'Running on set {set}')
                config.set = set
                DataHandler = dh.get_data(config, set)


                for train_type in train_types:
                    # start counting time
                    starting_time = time.time()

                    print(f'#################### Running {train_type} model #############################')
                    config.train_type = train_type
                    config.save_model = False
                    print(f'Running cross_v_HPS with {train_type} model')
                    if config.train_type == 'no_tl':
                        config.epochs = 0
                    else:
                        config.epochs = 100 # TODO was 100
                        opt_epochs = cv_tl.cross_v_HPS(config, DataHandler)
                        config.epochs = opt_epochs

                    epochs_chosen[train_type].append(config.epochs)

                    print(f'Running full_train with {train_type} model')
                    config.save_model = True
                    mean = cv_tl.train_6(config, DataHandler)

                    end_time = time.time()
                    train_times[train_type].append(end_time - starting_time)

                    print(f'Running ensemble with {train_type} model')
                    spearmanr = ensemble_util.train_ensemble(config, DataHandler)
                    testing_util.save_results(config, set, train_type, mean, spearmanr)
                    keras.backend.clear_session()

            for train_type in train_types:
                print(f'Average time for {train_type} model: {np.mean(train_times[train_type])}, using {dataset} dataset')
                print(f'std time for {train_type} model: {np.std(train_times[train_type])}, using {dataset} dataset')
                print(f'Average epochs chosen for {train_type} model: {np.mean(epochs_chosen[train_type])}, using {dataset} dataset')
                print(f'std epochs chosen for {train_type} model: {np.std(epochs_chosen[train_type])}, using {dataset} dataset')

            
        print('Full simulation done')


    if config.simulation_type == 'postprocess':
        postprocess.postprocess(config)
        print('Postprocessing done')