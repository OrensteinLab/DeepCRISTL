import pickle
import numpy as np

def get_data(config, set, verbose=1):
    if verbose > 0 :
        print('Reading data')


    dir_path = f'data/tl_train/{config.tl_data_category}/{config.tl_data}/set{set}/'



    with open(dir_path + 'test_seq.pkl', "rb") as fp:
        test_seq = pickle.load(fp)

    with open(dir_path + 'valid_seq.pkl', "rb") as fp:
        valid_seq = pickle.load(fp)

    with open(dir_path + 'train_seq.pkl', "rb") as fp:
        train_seq = pickle.load(fp)



    if config.model_type == 'cnn':
        convert_to_one_hot(test_seq)
        convert_to_one_hot(valid_seq)
        convert_to_one_hot(train_seq)


    DataHandler = {}

    if config.enzyme == 'multi_task':
        shape = train_seq.X_biofeat.shape[0]
        ohe_mat = np.repeat(np.array([[0.333, 0.333, 0.333]]), shape, axis=0)
        DataHandler['X_biofeat_train'] = np.concatenate((train_seq.X_biofeat, ohe_mat), axis=1)

        shape = valid_seq.X_biofeat.shape[0]
        ohe_mat = np.repeat(np.array([[0.333, 0.333, 0.333]]), shape, axis=0)
        DataHandler['X_biofeat_valid'] = np.concatenate((valid_seq.X_biofeat, ohe_mat), axis=1)

        shape = test_seq.X_biofeat.shape[0]
        ohe_mat = np.repeat(np.array([[0.333, 0.333, 0.333]]), shape, axis=0)
        DataHandler['X_biofeat_test'] = np.concatenate((test_seq.X_biofeat, ohe_mat), axis=1)

    else:
        DataHandler['X_biofeat_train'] = train_seq.X_biofeat
        DataHandler['X_biofeat_valid'] = valid_seq.X_biofeat
        DataHandler['X_biofeat_test'] = test_seq.X_biofeat

    DataHandler['X_train'] = train_seq.X
    # DataHandler['up_train'] = train_seq.up
    # DataHandler['down_train'] = train_seq.down
    # DataHandler['new_features_train'] = train_seq.new_features
    DataHandler['y_train'] = train_seq.y

    DataHandler['X_valid'] = valid_seq.X
    # DataHandler['up_valid'] = valid_seq.up
    # DataHandler['down_valid'] = valid_seq.down
    # DataHandler['new_features_valid'] = valid_seq.new_features
    DataHandler['y_valid'] = valid_seq.y

    DataHandler['X_test'] = test_seq.X
    # DataHandler['up_test'] = test_seq.up
    # DataHandler['down_test'] = test_seq.down
    # DataHandler['new_features_test'] = test_seq.new_features
    DataHandler['y_test'] = test_seq.y

    # if config.model_type == 'cnn':
    #     expand(test_seq)





    return DataHandler


def expand(data_seq):
    X_biofeat = data_seq.X_biofeat
    X_biofeat = X_biofeat.reshape((X_biofeat.shape[0], 11, 1))
    data_seq.X_biofeat = X_biofeat


def convert_to_one_hot(data_seq):
    X = data_seq.X[:, 1:] - 1
    X = np.eye(4)[X]
    X = X.reshape((X.shape[0], 21, 4, 1))
    data_seq.X = X