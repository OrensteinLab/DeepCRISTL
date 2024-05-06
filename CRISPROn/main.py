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
                        config.epochs = 100
                        opt_epochs = cv_tl.cross_v_HPS(config, DataHandler)
                        config.epochs = opt_epochs
                        # config.epochs = 100
                        # hps.param_search(config, DataHandler)

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
                print(f'Average time for {train_type} model: {sum(train_times[train_type]) / len(train_times[train_type])}, using {dataset} dataset')

            
        print('Full simulation done')


    if config.simulation_type == 'postprocess':
        postprocess.postprocess(config)
        print('Postprocessing done')