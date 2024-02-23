from scripts_tool import configurations as cfg
from scripts_tool import preprocess
from scripts_tool import data_handler as dh
from scripts_tool import cross_validation_tl as cv_tl
#from scripts_tool import ensemble_util
#from scripts_tool import  hyper_parameter_search as hps
import keras


if __name__ == '__main__':
    # Get configs
    config = cfg.get_parser()

    if config.action == 'new_data':
        if config.new_data_path == 'ALL_ORIGINAL_DATA':
            data_paths = ['leenay','chari2015Train293T','doench2014-Hs','doench2014-Mm','doench2016_hg19','hart2016-Hct1162lib1Avg','hart2016-HelaLib1Avg','hart2016-HelaLib2Avg','hart2016-Rpe1Avg','morenoMateos2015','xu2015TrainHl60','xu2015TrainKbm7']
        else:
            data_paths = [config.new_data_path]
        for data_path in data_paths:
            config.new_data_path = data_path
            print('Preprocessing new data - ', config.new_data_path)
            preprocess.prepare_inputs(config)

            print(f'Training on {config.new_data_path} dataset')
            train_types = ['LL_tl', 'gl_tl']

            DataHandler = dh.get_data(config)

            for train_type in train_types:
                print(f'#################### Running {train_type} model #############################')
                config.train_type = train_type
                config.save_model = False
                print(f'Running cross_v_HPS with {train_type} model')
                config.epochs = 100
                opt_epochs = cv_tl.cross_v_HPS(config, DataHandler)
                config.epochs = opt_epochs

                print(f'Running full_train with {train_type} model')
                config.save_model = True
                mean = cv_tl.train_6(config, DataHandler)

                keras.backend.clear_session()
            
            print('Training complete for ', config.new_data_path)


        

    


    if config.action == 'prediction':
        exit()

    if config.action == 'heat_map':
        exit()

    if config.action == 'saliency_maps':
        exit()
