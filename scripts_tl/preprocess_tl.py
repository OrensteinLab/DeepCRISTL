import os
import pandas as pd
import numpy as np
import math
import pickle
from scripts import feature_util
from sklearn.preprocessing import MinMaxScaler
import scipy as sp
from new_version import redistribute_split
from sklearn.model_selection import train_test_split


def char2int(char):
    if char == 'A':
        return 1
    if char == 'T':
        return 2
    if char == 'C':
        return 3
    if char == 'G':
        return 4
    else:
        print('Received wrong char {} - exiting'.format(char))
        exit(1)


def prepare_inputs(config):
    dir_path = f'data/tl_train/{config.tl_data_category}/{config.tl_data}/'

    if config.tl_data == 'leenay':
        if config.tl_data_category == 'random':
            print('Preparing random leenay')
            prepare_random_leenay(dir_path)
        else:
            prepare_leenay(dir_path)

    elif config.tl_data_category == 'U6T7':
        prepare_u6_t7_files(config, dir_path)

    elif config.tl_data_category == 'crispr_il':
        prepare_crispr_il_files(config, dir_path)

    else:
        print(f'Received wrong tl_data_category - {config.tl_data_category}')



    prepare_sequences(reads_sum=False, dir_path=dir_path,config=config,  old=False)



def prepare_crispr_il_files(config, dir_path):
    main_dataframes_path = 'data/main_dataframes/crispr_il/'

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if os.path.exists(dir_path + 'set4/train.csv'):
        return

    df = pd.read_csv(f'data/main_dataframes/crispr_il/{config.tl_data}.tsv', sep='\t')

    seq = df['sgRNA_seq_0_char']
    for i in range(1, 21):
        col = f'sgRNA_seq_{i}_char'
        seq = seq + df[col]



    new_df = df[['efficiency']]
    new_df.rename(columns={'efficiency': 'mean_eff'}, inplace=True)
    new_df['21mer'] = seq

    # down = df['downstream_seq_0_char']
    # for i in range(1, 20):
    #     col = f'downstream_seq_{i}_char'
    #     down = down + df[col]
    #
    # up = df['upstream_seq_0_char']
    # for i in range(1, 20):
    #     col = f'upstream_seq_{i}_char'
    #     up = up + df[col]
    #
    # new_df['downstream'] = down
    # new_df['upstream'] = up

    orig_size = new_df.shape[0]
    new_df.dropna(inplace=True)
    new_df.reset_index(drop=True, inplace=True)

    no_none_size = new_df.shape[0]
    print(f'There are {orig_size - no_none_size} samples with None value, left with {no_none_size} samples')

    feature_options = {
        "testing_non_binary_target_name": 'ranks',
        'include_pi_nuc_feat': True,
        "gc_features": True,
        "nuc_features": True,
        "include_Tm": True,
        "include_structure_features": True,
        "order": 3,
        "num_proc": 20,
        "normalize_features": None
    }
    feature_sets = feature_util.featurize_data(new_df, feature_options)
    for feature in feature_sets.keys():
        print(feature)
        reindexed_feature_df = feature_sets[feature]
        reindexed_feature_df.reset_index(inplace=True, drop=True)
        new_df = pd.concat([new_df, reindexed_feature_df], axis=1)


    new_df.to_csv(main_dataframes_path + f'{config.tl_data}.csv', index=False)

    choosen_bio = ['GC > 10', 'GC < 10', 'GC count', 'Tm global_False', '5mer_end_False', '8mer_middle_False',
                   '4mer_start_False', 'stem', 'dG', 'dG_binding_20', 'dg_binding_7to20']

    name_dict = {}
    for ind, bio_name in enumerate(choosen_bio):
        name_dict[bio_name] = f'epi{ind + 1}'

    new_df = new_df[['21mer', 'mean_eff'] + choosen_bio]
    new_df.rename(columns=name_dict, inplace=True)

    

    for i in range(5):
        print(f'Creating set {i} based on seed {i}')
        train_df, valid_df, test_df = redistribute_split.redistribute_tl_data(new_df, seed=i)
        train_val_df = pd.concat([train_df, valid_df], axis=0)

       
        perm_path = dir_path + f'set{i}/'
        os.mkdir(perm_path)
        test_df.to_csv(perm_path + 'test.csv', index=False)
        train_val_df.to_csv(perm_path + 'train_valid.csv', index=False)
        valid_df.to_csv(perm_path + 'valid.csv', index=False)
        train_df.to_csv(perm_path + 'train.csv', index=False)


    a=0



def prepare_u6_t7_files(config, dir_path):

    main_dataframes_path = 'data/main_dataframes/u6t7/'
    if not os.path.exists(main_dataframes_path):
        os.makedirs(main_dataframes_path)

        main_file = open('data/main_dataframes/13059_2016_1012_MOESM14_ESM.tsv')

        headers = main_file.readline()
        names = []
        for line in main_file.readlines():
            dataset_name = line.split('\t')[0]
            if dataset_name not in names:
                dataset_file = open(f'{main_dataframes_path}{dataset_name}.tsv', 'w')
                names.append(dataset_name)
                dataset_file.write(headers)
            dataset_file.write(line)
        dataset_file.close()

    if os.path.exists(dir_path + 'train.csv'):
        return
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if os.path.exists(main_dataframes_path + f'{config.tl_data}.csv'):
        df = pd.read_csv(main_dataframes_path + f'{config.tl_data}.csv')
    else:

        df = pd.read_csv(f'data/main_dataframes/u6t7/{config.tl_data}.tsv', sep='\t')
        df.rename(columns={'seq': '21mer'}, inplace=True)
        # df['21mer'] = df['21mer'].map(lambda x: x[0:21])

        # Check if all rows have longSeq100Bp as a string and print the ones that don't
        for index, row in df.iterrows():
            if type(row['longSeq100Bp']) != str:
                print(f'Row {index} is not a string but is a {type(row["longSeq100Bp"])}')
                print(row['longSeq100Bp'])

        df['21mer'] = df['longSeq100Bp'].map(lambda x: x[30:51])

        df['downstream'] = df['longSeq100Bp'].map(lambda x: x[6:30])
        df['upstream'] = df['longSeq100Bp'].map(lambda x: x[53:66])
        # mean_eff_col_name = config.mean_eff_col
        mean_eff_col_name = 'modFreq'

        # df_new = df[['21mer', 'downstream', 'upstream', mean_eff_col_name]]

        df.rename(columns={mean_eff_col_name: 'mean_eff'}, inplace=True)
        if config.tl_data in ['xu2015TrainHl60', 'xu2015TrainKbm7']:
            df.mean_eff = -df.mean_eff
        scaler = MinMaxScaler()
        df.mean_eff = scaler.fit_transform(df.mean_eff.to_numpy().reshape(-1, 1)).reshape(-1)
        # df_new.rename(columns={'wangOrig': 'mean_eff'}, inplace=True)
        # if (df_new['mean_eff'] > 10).any():
        #     df_new['mean_eff'] = df_new['mean_eff'] / 100
        feature_options = {
            "testing_non_binary_target_name": 'ranks',
            'include_pi_nuc_feat': True,
            "gc_features": True,
            "nuc_features": True,
            "include_Tm": True,
            "include_structure_features": True,
            "order": 3,
            "num_proc": 20,
            "normalize_features": None
        }
        columns = list(df.columns)
        feature_sets = feature_util.featurize_data(df, feature_options)
        for feature in feature_sets.keys():
            print(feature)
            reindexed_feature_df = feature_sets[feature]
            reindexed_feature_df.reset_index(inplace=True, drop=True)
            df = pd.concat([df, reindexed_feature_df], axis=1)

        choosen_bio = ['GC > 10', 'GC < 10', 'GC count', 'Tm global_False', '5mer_end_False', '8mer_middle_False',
                       '4mer_start_False', 'stem', 'dG', 'dG_binding_20', 'dg_binding_7to20']

        name_dict = {}
        for ind, bio_name in enumerate(choosen_bio):
            name_dict[bio_name] = f'epi{ind+1}'

        df = df[columns + choosen_bio]
        df.rename(columns=name_dict, inplace=True)
        df.to_csv(main_dataframes_path + f'{config.tl_data}.csv', index=False)

    split_data_set(df, dir_path)
    # test_df = df_new.sample(frac=0.2).sort_index()
    # train_val_df = df_new.drop(test_df.index)
    # merged = pd.merge(test_df, train_val_df, how='inner', on=['21mer']) #inner for intersection - to show that there are not same gRNA in the train and test set
    # assert merged.shape[0] == 0
    #
    #
    # valid_df = train_val_df.sample(frac=0.2).sort_index()
    # train_df = train_val_df.drop(valid_df.index)
    #
    # test_df.to_csv(dir_path + 'test.csv', index=False)
    # train_val_df.to_csv(dir_path + 'train_valid.csv', index=False)
    # valid_df.to_csv(dir_path + 'valid.csv', index=False)
    # train_df.to_csv(dir_path + 'train.csv', index=False)


def split_data_set(df, dir_path):
    for i in range(5):
        print(f'Creating set {i} based on seed {i}')
        train_df, valid_df, test_df = redistribute_split.redistribute_tl_data(df, seed=i)
        train_val_df = pd.concat([train_df, valid_df], axis=0)
        calc_spearman_for_comparison(test_df, dir_path, i)

       
        perm_path = dir_path + f'set{i}/'
        os.mkdir(perm_path)
        test_df.to_csv(perm_path + 'test.csv', index=False)
        train_val_df.to_csv(perm_path + 'train_valid.csv', index=False)
        valid_df.to_csv(perm_path + 'valid.csv', index=False)
        train_df.to_csv(perm_path + 'train.csv', index=False)

def calc_spearman_for_comparison(df, dir_path, set):
    results_path = dir_path + 'results.csv'
    comparison_expiriments = ['chariRank', 'chariRaw', 'crisprScan', 'doench', 'drsc', 'fusi', 'mh', 'oof', 'ssc', 'wang', 'wangOrig', 'wuCrispr']
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
    else:
        results_df = pd.DataFrame(columns=['set'] + comparison_expiriments)

    true_eff = df.mean_eff.to_numpy()
    result_dict = {'set': f'set{set}'}
    for exp in comparison_expiriments:
        exp_eff = df[exp].to_numpy()
        result_dict[exp] = sp.stats.spearmanr(exp_eff, true_eff)[0]

    results_df = results_df.append(result_dict, ignore_index=True)
    results_df.to_csv(results_path, index=False)

def prepare_leenay(dir_path):
    if os.path.exists(dir_path + 'set4/train.csv'):
        return
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    full_df = pd.read_csv('data/main_dataframes/leenay_full_data.csv')


    
    for i in range(5):
        print(f'Creating set {i} based on seed {i}')
        train_df, valid_df, test_df = redistribute_split.redistribute_tl_data(full_df, seed=i)
        train_val_df = pd.concat([train_df, valid_df], axis=0)

       
        perm_path = dir_path + f'set{i}/'
        os.mkdir(perm_path)
        test_df.to_csv(perm_path + 'test.csv', index=False)
        train_val_df.to_csv(perm_path + 'train_valid.csv', index=False)
        valid_df.to_csv(perm_path + 'valid.csv', index=False)
        train_df.to_csv(perm_path + 'train.csv', index=False)


def prepare_random_leenay(dir_path):
    if os.path.exists(dir_path + 'set4/train.csv'):
        return
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    full_df = pd.read_csv('data/main_dataframes/leenay_full_data.csv')


    
    for i in range(5):
        print(f'Creating set {i} based on seed {i}')
        train_val_df, test_df = train_test_split(full_df, test_size=0.2, random_state=i)
        train_df, valid_df = train_test_split(train_val_df, test_size=0.2, random_state=i)
        
        perm_path = dir_path + f'set{i}/'
        os.mkdir(perm_path)
        test_df.to_csv(perm_path + 'test.csv', index=False)
        train_val_df.to_csv(perm_path + 'train_valid.csv', index=False)
        valid_df.to_csv(perm_path + 'valid.csv', index=False)
        train_df.to_csv(perm_path + 'train.csv', index=False)


def prepare_sequences(reads_sum, dir_path, old, config, add_new_features=False):
    if os.path.exists(dir_path + 'set4/train_seq.pkl'):
        print('Sequence files already prepared -> returning')
        return
    dataframes = ['test', 'valid', 'train']

    for set in range(5):
        set_path = dir_path + f'set{set}/'
        for df_name in dataframes:
            df_path = set_path + df_name + '.csv'
            print(f'Preparing {df_path}')
            df = pd.read_csv(df_path)
            # downstream_size = len(df['downstream'][0])
            # upstream_size = len(df['upstream'][0])

            sequence = Seq()
            for index, row in df.iterrows():
                print('line: {}'.format(index))
                seq = row['21mer']
                up = None#row['upstream']
                down = None#row['downstream']

                biofeat = []
                for i in range(1, 12):
                    biofeat.append(row['epi{}'.format(i)])

                eff = row['mean_eff']
                # if config.tl_data in ['xu2015TrainHl60', 'xu2015TrainKbm7']:
                #     eff = -eff

                if add_new_features:
                    first_idx = df.columns.get_loc('epi11') + 1
                    new_features = row[first_idx:]
                    if index == 810:
                        continue
                    sequence.add_seq(seq, biofeat, up, down, eff, new_features)

                else:
                    sequence.add_seq(seq, biofeat, up, down, eff)


            with open(set_path + f'{df_name}_seq.pkl', "wb") as fp:
                pickle.dump(sequence, fp)



# Data structure objects


class Seq(object):
    # This class is the serialized data for one enzyme only
    def __init__(self, downstream_size=24, upstream_size=13):
        self.X = np.empty((0, 22), np.uint8)
        self.X_biofeat = np.empty((0, 11), np.float16)
        self.new_features = np.empty((0, 713), np.float16)
        self.up = np.empty((0, upstream_size, 4, 1), np.uint8)
        self.down = np.empty((0, downstream_size, 4, 1), np.uint8)
        self.y = np.empty(0, np.float16)
        self.confidence = np.empty(0, np.uint16)

        self.downstream_size = downstream_size
        self.upstream_size = upstream_size


    def add_seq(self, seq, biofeat,up, down, y, new_features=None):
        # The '0' represent the beginning of the sequence (help for the RNN)
        #mer21
        mer_array = np.array([0], dtype=np.uint8)
        for char in seq:
            num = char2int(char)
            mer_array = np.append(mer_array, np.array([num], dtype=np.uint8), axis=0)
        mer_array = np.expand_dims(mer_array, 0)
        self.X = np.concatenate((self.X, mer_array), axis=0)

        # Upstream
        if up != None:
            up_array = np.empty(shape=[0], dtype=np.uint8)
            for char in up:
                num = char2int(char)
                up_array = np.append(up_array, np.array([num], dtype=np.uint8), axis=0)
            up_array = up_array - 1
            up_array = np.eye(4)[up_array]
            up_array = up_array.reshape((self.upstream_size, 4, 1))
            up_array = np.expand_dims(up_array, 0)
            self.up = np.concatenate((self.up, up_array), axis=0)

        # Downstream
        if down != None:
            down_array = np.empty(shape=[0], dtype=np.uint8)
            for char in down:
                num = char2int(char)
                down_array = np.append(down_array, np.array([num], dtype=np.uint8), axis=0)
            down_array = down_array - 1
            down_array = np.eye(4)[down_array]
            down_array = down_array.reshape((self.downstream_size, 4, 1))
            down_array = np.expand_dims(down_array, 0)
            self.down = np.concatenate((self.down, down_array), axis=0)

        # biofeatures
        biofeat = np.array([float(epi) for epi in biofeat], dtype=np.float16)
        biofeat = np.expand_dims(biofeat, 0)
        self.X_biofeat = np.concatenate((self.X_biofeat,biofeat), axis=0)
        self.y = np.append(self.y, np.array([y], dtype=np.float16), axis=0)

        # New features
        if new_features is not None:
            new_features = np.array([float(epi) for epi in new_features], dtype=np.float16)
            new_features = np.expand_dims(new_features, 0)
            self.new_features = np.concatenate((self.new_features, new_features), axis=0)


