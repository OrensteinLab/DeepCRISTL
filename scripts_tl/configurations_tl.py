import argparse
from keras.optimizers import *

def get_parser():
    print('Receiving parser values')
    parser = argparse.ArgumentParser(description='Process some integers.')

    # Data
    parser.add_argument('--pre_train_data', type=str, default='DeepHF_old')  # [DeepHF_55k, DeepHF_full, DeepHF_old]
    parser.add_argument('--enzyme', type=str, default='multi_task')  # [wt,esp,hf,multi_task]
    parser.add_argument('--tl_data', type=str, default=None)  # [leenay]
    parser.add_argument('--tl_data_category', type=str, default=None)  # [leenay]
    parser.add_argument('--mean_eff_col', type=str, default=None)  # Supply the relevant column from the tsv file (Example-for xu/wang data we choose wangOrig)
    parser.add_argument('--transfer_learning','-tl', dest='transfer_learning', action='store_false')



    # Simulation
    parser.add_argument('-s_type', '--simulation_type', type=str, default=None)  # [train, param_search, cross_v_HPS, test_model, preprocess]


    # Training
    parser.add_argument('--train_type', type=str, default=None)  # [full_tl, LL_tl, no_em_tl, no_tl, gl_tl]



    # Model
    parser.add_argument('-mt', '--model_type', type=str, default='lstm')
    parser.add_argument('--add_flanks', dest='flanks', action='store_true')
    parser.add_argument('--add_new_features', dest='new_features', action='store_true')
    parser.add_argument('--dont_save_model', dest='save_model', action='store_false')
    parser.add_argument('--model_num', type=int, default=2)






    config = parser.parse_args()
    # Sanity check
    if config.simulation_type is None:
        print('No simulation type received >>>> exiting')
        exit(1)

    config = get_optimized_params(config)


    return config

def get_optimized_params(config):
    # This method will define the model optimized hyper parameters that were found in the Hyper Parameter search

    if config.pre_train_data == 'DeepHF_full':
        if config.enzyme == 'wt':
            config.batch_size = 60
            config.epochs = 14
            config.optimizer = Adamax
            config.init_lr = 0.002
            config.model_num = 0

        if config.enzyme == 'esp':
            config.batch_size = 100
            config.epochs = 16
            config.optimizer = Adamax
            config.init_lr = 0.002
            config.model_num = 0


    if config.pre_train_data == 'DeepHF_old':

        if config.enzyme == 'wt':
            config.batch_size = 100
            config.optimizer = RMSprop
            config.init_lr = 0.002
            config.model_num = 0

            if config.tl_data == 'leenay_anat':
                if config.train_type == 'full_tl':
                    config.epochs = 14
                elif config.train_type == 'LL_tl':
                    config.epochs = 5
                elif config.train_type == 'no_em_tl':
                    config.epochs = 14

            if config.tl_data == 'xu2015TrainHl60':
                if config.train_type == 'full_tl':
                    config.epochs = 30
                elif config.train_type == 'LL_tl':
                    config.epochs = 24
                elif config.train_type == 'no_em_tl':
                    config.epochs = 31
                elif config.train_type == 'gl_tl':
                    config.epochs = 49

            if config.tl_data == 'xu2015TrainKbm7':
                if config.train_type == 'full_tl':
                    config.epochs = 34
                elif config.train_type == 'LL_tl':
                    config.epochs = 19
                elif config.train_type == 'no_em_tl':
                    config.epochs = 31
                elif config.train_type == 'gl_tl':
                    config.epochs = 54

        if config.enzyme == 'esp':
            config.batch_size = 90
            config.epochs = 9
            config.optimizer = Adamax
            config.init_lr = 0.002
            config.model_num = 0
            if config.tl_data == 'leenay_anat':
                if config.train_type == 'full_tl':
                    config.epochs = 9
                elif config.train_type == 'LL_tl':
                    config.epochs = 13
                elif config.train_type == 'no_em_tl':
                    config.epochs = 10

            if config.tl_data == 'xu2015TrainHl60':
                if config.train_type == 'full_tl':
                    config.epochs = 63
                elif config.train_type == 'LL_tl':
                    config.epochs = 65
                elif config.train_type == 'no_em_tl':
                    config.epochs = 60
                elif config.train_type == 'gl_tl':
                    config.epochs = 2

        if config.enzyme == 'hf':
            config.batch_size = 110
            config.epochs = 41
            config.optimizer = Adamax
            config.init_lr = 0.002
            config.model_num = 0
            if config.tl_data == 'leenay_anat':
                if config.train_type == 'full_tl':
                    config.epochs = 41
                elif config.train_type == 'LL_tl':
                    config.epochs = 20

            if config.tl_data == 'xu2015TrainHl60':
                if config.train_type == 'full_tl':
                    config.epochs = 31
                elif config.train_type == 'LL_tl':
                    config.epochs = 32
                elif config.train_type == 'no_em_tl':
                    config.epochs = 33
                elif config.train_type == 'gl_tl':
                    config.epochs = 38


        if config.enzyme == 'multi_task':
            config.batch_size = 130
            config.optimizer = Adamax
            config.init_lr = 0.002
            config.model_num = 0
            if config.tl_data == 'leenay_anat':
                if config.train_type == 'full_tl':
                    config.epochs = 24
                elif config.train_type == 'LL_tl':
                    config.epochs = 18
                elif config.train_type == 'no_em_tl':
                    config.epochs = 22

            if config.tl_data == 'xu2015TrainHl60':
                if config.train_type == 'full_tl':
                    config.epochs = 69
                elif config.train_type == 'LL_tl':
                    config.epochs = 35
                elif config.train_type == 'no_em_tl':
                    config.epochs = 55
                elif config.train_type == 'gl_tl':
                    config.epochs = 74










    return config