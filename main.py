import tensorflow as tf
import numpy as np
import keras

# my scripts
from scripts import training_util
from scripts import configurations as cfg
from scripts import preprocess
from scripts import data_handler as dh
from scripts import fastq_preprocess
from scripts import models_util
from scripts import testing_util
from scripts import hyper_parameter_search as hps
from scripts import cross_validation as cv
from scripts import ensemble_util
from scripts import postprocess


'''standard command line:
     --data_type multi_task -s_type train -em --data_file final_df_1
     --enzyme multi_task -s_type train -ds new --model_type model3 --weighted_loss -md serial
     --enzyme esp -s_type train -ds new --model_type model3 --weighted_loss
     --enzyme esp -s_type param_search -ds new --model_type model1  -md parallel --weighted_loss
     --enzyme esp -s_type train/test_meand -ds new --model_type model3  -md serial --weighted_loss --model_path "model3/bio/esp" : for training and testing single enzyme
     --enzyme esp -s_type train -ds new --model_type model3 -tl --data_name leenay --model_path "model3/bio/esp" : for trasfer learning train
     --enzyme esp -s_type test --pre_train_data DeepHF_old  --model_type gl_lstm --layer_num 3 :gradual learning 
'''


if __name__ == '__main__':
    # f = open('C:/Softwares/sratoolkit.2.10.8-win64/bin/SRR14385344.fastq')
    # for i in range(10):
    #     line = f.readline()
    #     print(line)
    # Using only needed memory on GPU
    con = tf.compat.v1.ConfigProto()
    con.gpu_options.allow_growth = True
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=con))



    # Get configs
    config = cfg.get_parser()

    if config.simulation_type == 'fastq_preprocess':
        fastq_preprocess.create_data(config)

    if config.simulation_type == 'preprocess':
        preprocess.prepare_inputs(config)

    if config.simulation_type == 'train':
        # Initializing random seed
        np.random.seed(1234)

        print('simulation_type = train')

        print('Receiving DataHandler')
        DataHandler = dh.get_data(config)

        print('Building model')
        model, callback_list = models_util.get_model(config, DataHandler)
        history = training_util.train_model(config, DataHandler, model, callback_list)
        testing_util.test_model(config, model, DataHandler['test'])

    if config.simulation_type in ['cross_v', 'full_cross_v']:
        DataHandler = dh.get_data(config)
        cv.train_10_fold(config, DataHandler)


    if config.simulation_type == 'param_search':
        config.save_model = False
        DataHandler = dh.get_data(config)
        hps.param_search(config, DataHandler)

    if config.simulation_type == 'ensemble':
        DataHandler = dh.get_data(config)
        ensemble_util.train_ensemble(config, DataHandler)

    if config.simulation_type == 'postprocess':
        postprocess.postprocess(config)

    if config.simulation_type == 'test':
        DataHandler = dh.get_data(config)
        model_path = f'models/pre_train/{config.pre_train_data}/{config.enzyme}/'
        if config.model_type != 'lstm':
            model_path += f'{config.model_type}/'

        model_path += f'{config.model_name}/model'
        model = keras.models.load_model(model_path)
        testing_util.test_model(config, model, DataHandler['test'])










