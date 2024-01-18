import numpy as np
import os
import pickle
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import keras

# My Scripts
from scripts import models_util

from scripts_tl import training_util_tl
from scripts_tl import testing_util_tl


gl_init_lr = 0.002
def create_data(config, DataHandler):
    # Create main dir
    data_dir = f'data/tl_train/{config.tl_data_category}/{config.tl_data}/set{config.set}/10_fold/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if os.path.exists(data_dir + '9_fold/'):
        print('Data allready created')
        return


    X = np.concatenate((DataHandler['X_train'], DataHandler['X_valid']))
    X_biofeat = np.concatenate((DataHandler['X_biofeat_train'], DataHandler['X_biofeat_valid']))
    #print("TEST4")
    #print(X_biofeat.shape)
    y = np.concatenate((DataHandler['y_train'], DataHandler['y_valid']))


    perm = np.random.permutation(X.shape[0])
    val_size = int(X.shape[0] / 10)

    for ind in range(10):
        # Create fold dir
        fold_dir = data_dir + f'{ind}_fold/'
        os.mkdir(fold_dir)

        # Split val train
        valid_ind = perm[ind*val_size:(ind+1)*val_size]

        X_valid = X[valid_ind]
        X_train = np.delete(X, valid_ind, axis=0)

        X_biofeat_valid = X_biofeat[valid_ind]
        X_biofeat_train = np.delete(X_biofeat, valid_ind, axis=0)

        y_valid = y[valid_ind]
        y_train = np.delete(y, valid_ind, axis=0)


        pickle.dump(X_valid, open(fold_dir + f'X_valid.pkl', "wb"))
        pickle.dump(X_train, open(fold_dir + f'X_train.pkl', "wb"))
        pickle.dump(X_biofeat_valid, open(fold_dir + f'X_biofeat_valid.pkl', "wb"))
        pickle.dump(X_biofeat_train, open(fold_dir + f'X_biofeat_train.pkl', "wb"))
        pickle.dump(y_valid, open(fold_dir + f'y_valid.pkl', "wb"))
        pickle.dump(y_train, open(fold_dir + f'y_train.pkl', "wb"))




def load_fold_data(config, DataHandler, k):
    data_dir =  f'data/tl_train/{config.tl_data_category}/{config.tl_data}/set{config.set}/10_fold/' if config.transfer_learning else f'data/pre_train/{config.pre_train_data}/{config.enzyme}_10_fold/'
    data_dir += f'{k}_fold/'

    if config.enzyme == 'multi_task':
        DataHandler['X_valid'] = pickle.load(open(data_dir + 'X_valid.pkl', "rb"))
        DataHandler['X_train'] = pickle.load(open(data_dir + 'X_train.pkl', "rb"))

        DataHandler['X_biofeat_valid'] = pickle.load(open(data_dir + 'X_biofeat_valid.pkl', "rb"))
        DataHandler['X_biofeat_train'] = pickle.load(open(data_dir + 'X_biofeat_train.pkl', "rb"))
        shape = DataHandler['X_biofeat_valid'].shape[0]
        #ohe_mat = np.repeat(np.array([[0.333, 0.333, 0.333]]), shape, axis=0)
        #DataHandler['X_biofeat_valid'] = np.concatenate((DataHandler['X_biofeat_valid'], ohe_mat), axis=1) TODO: Removed for now
        shape = DataHandler['X_biofeat_train'].shape[0]
        #ohe_mat = np.repeat(np.array([[0.333, 0.333, 0.333]]), shape, axis=0)
        #DataHandler['X_biofeat_train'] = np.concatenate((DataHandler['X_biofeat_train'], ohe_mat), axis=1) TODO: Removed for now

        DataHandler['y_valid'] = pickle.load(open(data_dir + 'y_valid.pkl', "rb"))
        DataHandler['y_train'] = pickle.load(open(data_dir + 'y_train.pkl', "rb"))
    else:
        DataHandler['X_valid'] = pickle.load(open(data_dir + 'X_valid.pkl', "rb"))
        DataHandler['X_train'] = pickle.load(open(data_dir + 'X_train.pkl', "rb"))
        DataHandler['X_biofeat_valid'] = pickle.load(open(data_dir + 'X_biofeat_valid.pkl', "rb"))
        DataHandler['X_biofeat_train'] = pickle.load(open(data_dir + 'X_biofeat_train.pkl', "rb"))
        DataHandler['y_valid'] = pickle.load(open(data_dir + 'y_valid.pkl', "rb"))
        DataHandler['y_train'] = pickle.load(open(data_dir + 'y_train.pkl', "rb"))

    return DataHandler


def cross_v_HPS(config, DataHandler):
    
    #print("TEST3")
    #print(DataHandler['X_biofeat_train'].shape)
    #print(DataHandler['X_biofeat_train'])
    create_data(config, DataHandler)
    best_epoch_arr = []
    # config.epochs = 100
    for k in range(10):
        print(f'\nStarting training {k}')
        keras.backend.clear_session()
        DataHandler = load_fold_data(config, DataHandler, k)
        config.model_num = k
        if config.train_type in ['full_tl', 'LL_tl', 'gl_tl', 'no_em_tl', 'no_tl', 'no_pre_train']:
            if config.train_type == 'gl_tl':
                # TODO - use higher lr with multi_task
                config.init_lr = gl_init_lr

            model, callback_list = models_util.load_pre_train_model(config, DataHandler)
        else:
            model, callback_list = models_util.get_model(config, DataHandler)

        #print("TEST2")
        #print(DataHandler['X_biofeat_train'].shape)
        #print(DataHandler['X_biofeat_train'])

        history = training_util_tl.train_model(config, DataHandler, model, callback_list)

        best_epoch = history.history['val_loss'].index(min(history.history['val_loss'])) + 1

        best_epoch_arr.append(best_epoch)


    results_path = f'HPS/transfer_learning/{config.tl_data_category}/{config.tl_data}/set{config.set}/{config.pre_train_data}/{config.enzyme}/'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    results_path += f'cross_v_HPS_{config.train_type}.txt'
    f = open(results_path, 'w')
    f.write(f'best_epoch_arr: {best_epoch_arr}, mean: {np.mean(best_epoch_arr)}')


    f.close()
    opt_epochs = round(np.mean(best_epoch_arr))
    return opt_epochs


def train_10(config, DataHandler):
    spearman_result = []
    DataHandler['X_train'] = np.concatenate((DataHandler['X_train'], DataHandler['X_valid']))
    DataHandler['X_biofeat_train'] = np.concatenate((DataHandler['X_biofeat_train'], DataHandler['X_biofeat_valid']))
    DataHandler['y_train'] = np.concatenate((DataHandler['y_train'], DataHandler['y_valid']))

    for k in range(10):
        print(f'\nStarting training {k}')
        config.model_num = k
        keras.backend.clear_session()
        if config.train_type in ['full_tl', 'LL_tl', 'gl_tl', 'no_em_tl', 'no_tl', 'no_pre_train']:
            if config.train_type == 'gl_tl':
                config.init_lr = gl_init_lr
            model, callback_list = models_util.load_pre_train_model(config, DataHandler)
            # print model.summary()
            #model.summary()
        else:
            model, callback_list = models_util.get_model(config, DataHandler)

        history = training_util_tl.train_model(config, DataHandler, model, callback_list)
        spearman_result.append(testing_util_tl.test_model(config, model, DataHandler))

    results_path = f'results/transfer_learning/{config.tl_data}/set{config.set}/'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    results_path += f'{config.pre_train_data}_{config.enzyme}_{config.train_type}_10_mean.txt'
    f = open(results_path, 'w')

    res = dict()
    for dic in spearman_result:
        for list in dic:
            if list in res:
                res[list].append(dic[list])
            else:
                res[list] = [dic[list]]


    for enzyme, spearmans in res.items():
        mean = sum(spearmans) / len(spearmans)
        f.write(f'{enzyme}: spearmans - {spearmans}, mean - {mean}\n')

    f.close()
    return mean


def data_size_scan(config, DataHandler):
    dir_path = f'results/transfer_learning/{config.tl_data}/'
    if not os.path.exists:
        os.makedirs(dir_path)
    path = f'{dir_path}size_scan.pkl'

    if os.path.exists(path):
        plot_size_scan(config, dir_path)
        return

    X_train = np.concatenate((DataHandler['X_train'], DataHandler['X_valid']))
    X_biofeat_train = np.concatenate((DataHandler['X_biofeat_train'], DataHandler['X_biofeat_valid']))
    y_train = np.concatenate((DataHandler['y_train'], DataHandler['y_valid']))

    max_size = (round(X_train.shape[0] / 100) ) * 100
    size_spearman_results = {}

    for size in range(100, max_size, 100):
        print(f'Training size={size}')


        print('Searching for optimal number of epochs')
        config.epochs = 1000
        best_epoch_arr = []
        for k in range(0):
            start = time.time()

            keras.backend.clear_session()
            # Getting random data for this size
            perm = np.random.permutation(X_train.shape[0])
            indexes = perm[:size]
            train_indexes = indexes[:int(size * .9)]
            valid_indexes = indexes[int(size * .9):]

            DataHandler['X_train'] = X_train[train_indexes]
            DataHandler['X_biofeat_train'] = X_biofeat_train[train_indexes]
            DataHandler['y_train'] = y_train[train_indexes]

            DataHandler['X_valid'] = X_train[valid_indexes]
            DataHandler['X_biofeat_valid'] = X_biofeat_train[valid_indexes]
            DataHandler['y_valid'] = y_train[valid_indexes]

            print(f'Optimal epoch search: Size={size}, Train={k}')
            config.model_num = k
            if config.train_type in ['full_tl', 'LL_tl', 'gl_tl', 'no_em_tl']:
                if config.train_type == 'gl_tl':
                    config.init_lr = gl_init_lr
                model, callback_list = models_util.load_pre_train_model(config, DataHandler, verbose=0)
            else:
                model, callback_list = models_util.get_model(config, DataHandler, verbose=0)

            history = training_util_tl.train_model(config, DataHandler, model, callback_list, verbose=0)
            best_epoch = int(np.where(history.history['val_loss'] == np.amin(history.history['val_loss']))[0]) + 1
            end = time.time()

            print(f'Best epoch={best_epoch}, time={end-start}')
            best_epoch_arr.append(best_epoch)


        avg_best_epoch = round(sum(best_epoch_arr) / len(best_epoch_arr))


        print(f'Average epochs for size {size} is {avg_best_epoch}')
        config.epochs = avg_best_epoch
        print(f'Caluculating average spearman score for size {size}')
        spearman_results = []
        for k in range(10):
            start = time.time()
            keras.backend.clear_session()
            perm = np.random.permutation(X_train.shape[0])
            indexes = perm[:size]
            DataHandler['X_train'] = X_train[indexes]
            DataHandler['X_biofeat_train'] = X_biofeat_train[indexes]
            DataHandler['y_train'] = y_train[indexes]

            DataHandler['X_valid'] = X_train[indexes]
            DataHandler['X_biofeat_valid'] = X_biofeat_train[indexes]
            DataHandler['y_valid'] = y_train[indexes]

            print(f'Spearman calc: Size={size}, Train={k}')
            config.model_num = k
            if config.train_type in ['full_tl', 'LL_tl', 'gl_tl', 'no_em_tl']:
                if config.train_type == 'gl_tl':
                    config.init_lr = gl_init_lr
                model, callback_list = models_util.load_pre_train_model(config, DataHandler, verbose=0)
            else:
                model, callback_list = models_util.get_model(config, DataHandler, verbose=0)
            history = training_util_tl.train_model(config, DataHandler, model, callback_list, verbose=0)
            spearman = testing_util_tl.test_model(config, model, DataHandler, verbose=0)
            end = time.time()
            print(f'Spearman={spearman}, time={end - start}')
            spearman_results.append(spearman)

        res = dict()
        for dic in spearman_results:
            for list in dic:
                if list in res:
                    res[list].append(dic[list])
                else:
                    res[list] = [dic[list]]

        spearmans = res[config.enzyme]
        mean = sum(spearmans) / len(spearmans)
        size_spearman_results[size] = mean
        print(f'Size={size} Average spearman={mean}')
        a=0

    with open(path, "wb") as fp:
        pickle.dump(size_spearman_results, fp)

    plot_size_scan(config, dir_path)
    a=0

def plot_size_scan(config, dir_path):
    path = f'{dir_path}size_scan.pkl'
    # size_spearman_results = {100: 0.16718866666182805, 200:0.23268288650353589, 300:0.2649458397090809,
    #                          400:0.2725523083779643, 500:0.302700281532319, 600:0.30575089387658333,
    #                          700:0.31973598033673045, 800:0.3294480538207073, 900:0.328821478931025,
    #                          1000:0.3342742667368041, 1100:0.3475471199679201}

    # size_spearman_results = {0:[0.197738335530377, 0.2068747697651391,  0.19471378966294373, 0.1904340835528094, 0.18485179598510829, 0.22068193973716693, 0.19688023306695712, 0.2426903101048743, 0.22820281555895114,  0.20855793617169383],
    #                         100: [0.2580189027949475, 0.14367661170458787, 0.11837791332692395, 0.14602936889340123, 0.03295914074850166, 0.12947241723206046 , 0.11716523365565641, 0.30262529998028204, 0.2866566901416776, 0.13690508814024188],
    #                         200:[0.2611826890847193, 0.281143197173753, 0.12217620583409546, 0.2212537415789342, 0.11486291157643123, 0.2774777035640053, 0.2335686531156784, 0.2555469723229943, 0.30136686264910584, 0.25824992813564174],
    #                         300:[0.24232361871153066, 0.28561219529149645, 0.1698054557166831, 0.2967331369556913, 0.2362028707547742, 0.21616508306233914, 0.3159649797316022, 0.31909805754739085, 0.303188830982979, 0.26436416833632237],
    #                          400:[0.21248230322551478, 0.2895882308864909, 0.1942109645729305, 0.2945477511184549, 0.3655282533729949, 0.27320393812853866, 0.24317572802271817, 0.3025219893525772, 0.24311553127901614, 0.3071483938204063],
    #                          500:[0.2804218530862156, 0.29021805965421105, 0.3016265627900095, 0.31064366893529394, 0.29066912849722154, 0.2925429555123245, 0.30262794375618784, 0.303164223530317, 0.276238382766302, 0.3788500367951064],
    #                          600:[0.3373832251816365, 0.23181827009864003,  0.33852147239278596, 0.3311085281202096, 0.3268408637060642, 0.28842110550775457, 0.31276397721177146, 0.2779181973033924, 0.31371288939458786, 0.2990204098489909],
    #                          700:[0.3559457859186171, 0.3100774941566911, 0.3107036623116186, 0.30478160428255424, 0.3047338129488719, 0.28037019777236316, 0.3367387539627455, 0.3413228580160813, 0.3505770905233812, 0.3021085434743805],
    #                          800:[0.31590071564035277, 0.2998540127288376, 0.31590417288576805, 0.3304624299622997, 0.35480021748188223, 0.34004855802947015, 0.2824571537989517, 0.35681680839590013, 0.3439918514767077, 0.3542446178069027],
    #                          900:[0.32001727809311104, 0.33974737094358265, 0.3378273795338163, 0.33626836521888476, 0.3705774586183798, 0.3052749735400581, 0.26314782805226217, 0.3518540342858982, 0.31644513010957687, 0.3470549709146804],
    #                          1000:[0.32487226749310594, 0.325775015281259, 0.2850088042828346, 0.36577412453223734, 0.39041736985263215, 0.34763761845085045, 0.28104476736310513, 0.35505524017310636, 0.3504390040741458, 0.31671845586476444 ],
    #                          1100:[0.32478298921443977, 0.32246317754076065, 0.34150568528833847, 0.3199013586880091, 0.3638193573009404, 0.3369803544070631, 0.34814440995526075, 0.38858116580234303, 0.33692402164353114, 0.39236867983851376]}
    #
    #
    # with open(path, "wb") as fp:
    #     pickle.dump(size_spearman_results, fp)

    with open(path, "rb") as fp:
        size_spearman_results = pickle.load(fp)

    fig = plt.figure
    data = list(size_spearman_results.items())
    an_array = np.array(data)
    sizes = an_array[:,0]
    spearmans = an_array[:,1]
    size_spearman_arr = []
    size_spearman_std_arr = []
    for spearman in spearmans:
        mean_s = sum(spearman) / len(spearman)
        size_spearman_arr.append(mean_s)

        std = np.std(spearman)
        size_spearman_std_arr.append(std)
        a=0
    # plt.plot(sizes, spearmans)
    plt.errorbar(range(len(sizes)), size_spearman_arr, yerr=size_spearman_std_arr, fmt='--o', capthick=10)
    plt.xlabel('Transfer train data samples')
    plt.ylabel('Spearman coorelation')
    plt.title('Data: Leenay, Data size VS. Spearman correlation')
    plt.savefig(dir_path + 'size_scan_plot.png')



