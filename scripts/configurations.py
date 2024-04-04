import argparse
from keras.optimizers import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer

'''
Model Types:
    model1 - This is the original model, or the mixed model when 'enzyme'='multi_task'
            Inputs: Sequence + biofetures
            Output: Efficiency per enzyme
            Architecture: embeding -> lstm -> FC
              
    model2 - This model is using only the biofeatures as inputs and have only FC layers
            Inputs: biofetures
            Output: Efficiency per enzyme
            Architecture: FC
'''
def get_parser():
    print('Receiving parser values')
    parser = argparse.ArgumentParser(description='Process some integers.')

    # Data
    parser.add_argument('--enzyme', type=str, default=None)  # [wt,esp,hf,multi_task]
    parser.add_argument('--data_file', type=str, default='final_efficiency_with_bio')  # original file = 'row_data'
    parser.add_argument('-ds', '--data_source', type=str, default='new')  # For original data choose old  ['old', 'new']
    parser.add_argument('-md', '--multi_data', type=str, default='parallel')  # ['parallel', 'serialized] this option is for multi task only
    parser.add_argument('--transfer_learning','-tl', dest='transfer_learning', action='store_true')
    parser.add_argument('--no_weights', dest='no_weights', action='store_true')
    parser.add_argument('--data_name', type=str, default=None)  # Relevant only for transfer learning
    parser.add_argument('--pre_train_data', type=str, default='DeepHF_old')  # [DeepHF_55k, DeepHF_full, DeepHF_old]





    # Simulation
    parser.add_argument('-s_type', '--simulation_type', type=str, default=None)  # [train, param_search, cross_v, test_model, preprocess]
    parser.add_argument('--model_path', type=str, default='None')  # For the test_model option
    parser.add_argument('--eda', dest='eda', action='store_true')  # For the test_model option


    # Training
    parser.add_argument('--lr_scheduler', dest='lr_scheduler', action='store_true')


    # Model
    parser.add_argument('--dont_save_model', dest='save_model', action='store_false')
    parser.add_argument('-no_em', '--has_embedding', dest='has_embedding', action='store_false')
    parser.add_argument('-no_bio', '--no_biofeatures', dest='has_biofeatures', action='store_false')
    parser.add_argument('-mt', '--model_type', type=str, default='lstm')
    parser.add_argument('--weighted_loss', dest='weighted_loss', action='store_true')  # weighted_loss / row_reads model
    parser.add_argument('--model_name', type=str, default='model_1') #TODO was 2???


    # GL model
    parser.add_argument('--layer_num', type=int, default=None)






    config = parser.parse_args()
    # Sanity check
    if config.enzyme is None and config.simulation_type != 'preprocess' and config.simulation_type != 'postprocess':
        print('No data type received >>>> exiting')
        exit(1)

    if config.simulation_type is None:
        print('No simulation type received >>>> exiting')
        exit(1)

    config = get_optimized_params(config)

    if config.lr_scheduler:
        config.learning_rate = 0.01
    return config

def get_optimized_params(config):
    # This method will define the model optimized hyper parameters that were found in the Hyper Parameter search

    if config.pre_train_data == 'DeepHF_full':
        if config.enzyme == 'esp':
            config.em_dim = 71
            config.em_drop = 0.2

            config.rnn_drop = 0.5
            config.rnn_rec_drop = 0.4
            config.rnn_units = 90

            config.fc_num_hidden_layers = 2
            config.fc_num_units = 300
            config.fc_drop = 0.4
            config.fc_activation = 'hard_sigmoid'

            config.batch_size = 100
            config.epochs = 100
            config.optimizer = Adamax
            config.last_activation = 'sigmoid'
            config.initializer = 'he_normal'


        if config.enzyme == 'hf':
            config.em_dim = 36
            config.em_drop = 0.3

            config.rnn_drop = 0.5
            config.rnn_rec_drop = 0.7
            config.rnn_units = 210

            config.fc_num_hidden_layers = 1
            config.fc_num_units = 110
            config.fc_drop = 0.2
            config.fc_activation = 'relu'

            config.batch_size = 110
            config.epochs = 100
            config.optimizer = RMSprop
            config.last_activation = 'sigmoid'
            config.initializer = 'normal'


        if config.enzyme == 'wt':
            config.em_dim = 67
            config.em_drop = 0.5

            config.rnn_drop = 0.3
            config.rnn_rec_drop = 0.1
            config.rnn_units = 110

            config.fc_num_hidden_layers = 2
            config.fc_num_units = 280
            config.fc_drop = 0.7
            config.fc_activation = 'sigmoid'

            config.batch_size = 60
            config.epochs = 100
            config.optimizer = Adamax
            config.last_activation = 'linear'
            config.initializer = 'he_normal'


    if config.pre_train_data == 'DeepHF_55k':
        if config.enzyme == 'esp':
            config.em_dim = 35
            config.em_drop = 0.6

            config.rnn_drop = 0.3
            config.rnn_rec_drop = 0.1
            config.rnn_units = 80

            config.fc_num_hidden_layers = 3
            config.fc_num_units = 280
            config.fc_drop = 0.6
            config.fc_activation = 'sigmoid'

            config.batch_size = 120
            config.epochs = 100
            config.optimizer = Adam
            config.last_activation = 'sigmoid'
            config.initializer = 'lecun_uniform'

        if config.enzyme == 'wt':
            config.em_dim = 78
            config.em_drop = 0.4

            config.rnn_drop = 0.8
            config.rnn_rec_drop = 0.1
            config.rnn_units = 200

            config.fc_num_hidden_layers = 1
            config.fc_num_units = 230
            config.fc_drop = 0.5
            config.fc_activation = 'hard_sigmoid'

            config.batch_size = 130
            config.epochs = 100
            config.optimizer = Adamax
            config.last_activation = 'linear'
            config.initializer = 'he_normal'


        if config.enzyme == 'hf':
            config.em_dim = 79
            config.em_drop = 0.3

            config.rnn_drop = 0.1
            config.rnn_rec_drop = 0.6
            config.rnn_units = 70

            config.fc_num_hidden_layers = 1
            config.fc_num_units = 150
            config.fc_drop = 0.6
            config.fc_activation = 'elu'

            config.batch_size = 80
            config.epochs = 100
            config.optimizer = Nadam
            config.last_activation = 'linear'
            config.initializer = 'he_normal'


    if config.pre_train_data == 'DeepHF_old':
        if config.enzyme == 'multi_task':
            config.em_dim = 43
            config.em_drop = 0.1

            config.rnn_drop = 0.1
            config.rnn_rec_drop = 0.5
            config.rnn_units = 220

            config.fc_num_hidden_layers = 3
            config.fc_num_units = 190
            config.fc_drop = 0.4
            config.fc_activation = 'elu'

            config.batch_size = 130
            config.epochs = 100
            config.optimizer = Adamax
            config.last_activation = 'sigmoid'
            config.initializer = 'he_uniform'

        if config.enzyme == 'esp':
            config.em_dim = 36
            config.em_drop = 0.5

            config.rnn_drop = 0.4
            config.rnn_rec_drop = 0.4
            config.rnn_units = 130

            config.fc_num_hidden_layers = 1
            config.fc_num_units = 90
            config.fc_drop = 0.4
            config.fc_activation = 'relu'

            config.batch_size = 100
            config.epochs = 100
            config.optimizer = RMSprop
            config.last_activation = 'sigmoid'
            config.initializer = 'he_uniform'

        if config.enzyme == 'wt':
            config.em_dim = 63
            config.em_drop = 0.8

            config.rnn_drop = 0.1
            config.rnn_rec_drop = 0.2
            config.rnn_units = 110

            config.fc_num_hidden_layers = 3
            config.fc_num_units = 220
            config.fc_drop = 0.6
            config.fc_activation = 'sigmoid'

            config.batch_size = 90
            config.epochs = 100
            config.optimizer = Adamax
            config.last_activation = 'linear'
            config.initializer = 'normal'

        if config.enzyme == 'hf':
            config.em_dim = 67
            config.em_drop = 0.1

            config.rnn_drop = 0.3
            config.rnn_rec_drop = 0.3
            config.rnn_units = 190

            config.fc_num_hidden_layers = 3
            config.fc_num_units = 280
            config.fc_drop = 0.5
            config.fc_activation = 'elu'

            config.batch_size = 110
            config.epochs = 100
            config.optimizer = Adamax
            config.last_activation = 'linear'
            config.initializer = 'he_normal'








    return config