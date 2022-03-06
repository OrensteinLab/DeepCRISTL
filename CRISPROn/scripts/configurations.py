import argparse
from keras.optimizers import *

def get_parser():
    print('Receiving parser values')
    parser = argparse.ArgumentParser(description='Process some integers.')

    # Data
    parser.add_argument('--tl_data', type=str, default=None)  # [leenay]
    parser.add_argument('--tl_data_category', type=str, default=None)  # [leenay]

    # Simulation
    parser.add_argument('-s_type', '--simulation_type', type=str, default=None)  # [train, param_search, cross_v_HPS, test_model, preprocess]

    # Model
    parser.add_argument('--dont_save_model', dest='save_model', action='store_false')
    parser.add_argument('-t_type', '--train_type', type=str, default=None)

    # Hyper parameters
    parser.add_argument('--init_lr', type=float, default=0.0043)


    config = parser.parse_args()

    config.optimizer = RMSprop
    config.batch_size = 80

    # Sanity check
    if config.simulation_type is None:
        print('No simulation type received >>>> exiting')
        exit(1)


    return config