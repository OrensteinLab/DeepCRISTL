from scripts import configurations as cfg
from scripts import preprocess
from scripts import data_handler as dh
from scripts import cross_validation_tl as cv_tl
from scripts import ensemble_util
from scripts import testing_util
from scripts import  hyper_parameter_search as hps
from scripts import postprocess
import keras

#interperter = ModelInterpertation

if __name__ == '__main__':
    # Get configs
    config = cfg.get_parser()

    if config.simulation_type == 'preprocess':
        preprocess.prepare_inputs(config)
        print('Preprocessing done')

    if config.simulation_type == 'full_sim':
        print(f'Running full simulation for {config.tl_data} dataset')

        train_types = ['full_tl', 'LL_tl', 'gl_tl', 'no_tl', 'no_pre_train']
        for set in range(5):
            print(f'Running on set {set}')
            config.set = set
            DataHandler = dh.get_data(config, set)


            for train_type in train_types:
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

                print(f'Running ensemble with {train_type} model')
                spearmanr = ensemble_util.train_ensemble(config, DataHandler)

                testing_util.save_results(config, set, train_type, mean, spearmanr)
                keras.backend.clear_session()

            
        print('Full simulation done')


    if config.simulation_type == 'postprocess':
        postprocess.postprocess(config)
        print('Postprocessing done')