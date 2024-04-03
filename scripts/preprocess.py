import pandas as pd
import os
import numpy as np
import pickle
from scripts import feature_util
from new_version import redistribute_split
import math

# def create_dataframes(config):
#     if config.transfer_learning:
#         a=0
#     else:
#         if config.data_source == 'new':
#             full_df = pd.read_csv('data/main_dataframes/final_efficiency_with_bio.csv')
#             full_df = full_df.iloc[:20]
#             split_enzymes_dataframe(full_df, row_reads=True)
#         elif config.data_source == 'old':
#             full_df = pd.read_csv('data/main_dataframes/supplementary2_with_bio.csv')
#             split_enzymes_dataframe(full_df, row_reads=False)
#
#             a=0
#         a=0
#     a=0
#
# def split_enzymes_dataframe(full_df, enzymes, row_reads):
#     # there are two gRNA which are the same, we will drop the duplicates
#     full_df = full_df.drop_duplicates(subset=['21mer'], keep=False)
#
#     # For the test file we use only reads with threshold >= 100
#     wt_cond = full_df.wt_reads_sum >= 100
#     esp_cond = full_df.esp_reads_sum >= 100
#     hf_cond = full_df.hf_reads_sum >= 100
#     df_triple_positive = full_df[(full_df['wt_reads_sum'] >= 100) & (full_df['esp_reads_sum'] >= 100) & (full_df['hf_reads_sum'] >= 100)]
#
#     # The test file need to be the same for all models
#     test_df = full_df[wt_cond & esp_cond & hf_cond].sample(frac=0.15).sort_index()
#     train_val_full_df = full_df.drop(test_df.index)
#     train_val_55K_df = full_df[wt_cond | esp_cond | hf_cond].drop(test_df.index)
#
#
#     # Prepare test - Add one hot encoding and divide by enzyme
#     # 55k data (100 reads threshold)
#     dir_path = 'data/pre_train/DeepHF_55k/'
#     if not os.path.exists(dir_path):
#         os.mkdir(dir_path)
#
#     valid_55k_df = train_val_55K_df.sample(frac=0.1).sort_index()
#     train_55k_df = train_val_55K_df.drop(valid_55k_df.index)




######################################################################################
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
    choosen_bio = ['GC > 10', 'GC < 10', 'GC count', 'Tm global_False', '5mer_end_False', '8mer_middle_False',
                   '4mer_start_False', 'stem', 'dG', 'dG_binding_20', 'dg_binding_7to20']

    data_columns = ['gRNA', '21mer']
    enzymes = ['wt', 'hf', 'esp']
    for enzyme in enzymes:
        data_columns += [f'{enzyme}_reads_sum', f'{enzyme}_edited_read_counts', f'{enzyme}_mean_eff']
    data_columns += choosen_bio

    if config.transfer_learning:
        a=0 # TODO
    else:
        if config.pre_train_data == 'DeepHF_old':
            data_columns = ['21mer', 'Wt_Efficiency', 'eSpCas 9_Efficiency', 'SpCas9-HF1_Efficiency'] + choosen_bio
            prepare_old_data(data_columns)
            prepare_sequences(reads_sum=False, dir_path='data/pre_train/DeepHF_old/', old=True)

        else:
            prepare_dataframes(data_columns=data_columns)
            prepare_sequences(reads_sum=False, dir_path='data/pre_train/DeepHF_55k/', old=False)
            prepare_sequences(reads_sum=True, dir_path='data/pre_train/DeepHF_full/', old=False)



def prepare_old_data(data_columns):
    dir_path = 'data/pre_train/DeepHF_old/'
    if os.path.exists(dir_path + 'train.csv'):
        return
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if os.path.exists('data/main_dataframes/supplementary2_with_bio.csv'):
        full_df = pd.read_csv('data/main_dataframes/supplementary2_with_bio.csv')
    else:
        df = pd.read_csv('data/main_dataframes/dhf_pretrain.csv')

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
        feature_sets = feature_util.featurize_data(df, feature_options)


        for feature in feature_sets.keys():
            print(feature)
            reindexed_feature_df = feature_sets[feature]
            reindexed_feature_df.reset_index(inplace=True, drop=True)
            df = pd.concat([df, reindexed_feature_df], axis=1)
        df.to_csv('data/main_dataframes/supplementary2_with_bio.csv', index=False)
        full_df = df

    full_df = full_df[data_columns]
    name_dict = {'Wt_Efficiency': 'wt_mean_eff', 'eSpCas 9_Efficiency': 'esp_mean_eff', 'SpCas9-HF1_Efficiency': 'hf_mean_eff'}
    for ind, bio_name in enumerate(data_columns[4:]):
        name_dict[bio_name] = f'epi{ind+1}'
    full_df.rename(columns=name_dict, inplace=True)

    train_df, val_df, test_df = redistribute_split.redistribute_dhf_pretrain_data(full_df)


    train_val_df = pd.concat([train_df, val_df], axis=0)

    test_df.to_csv(dir_path + 'test.csv', index=False)
    train_val_df.to_csv(dir_path + 'train_valid.csv', index=False)
    val_df.to_csv(dir_path + 'valid.csv', index=False)
    train_df.to_csv(dir_path + 'train.csv', index=False)


def prepare_dataframes(data_columns):
    if not os.path.exists('data/pre_train'):
        os.mkdir('data/pre_train')

    if os.path.exists('data/pre_train/DeepHF_55k/train.csv'):
        print('Dataframes already been created')
        return

    full_df = pd.read_csv('fastq_files/efficiency/final_efficiency_with_bio.csv')
    full_df = full_df[data_columns]

    # Convert biofeatures names to general names
    name_dict = {}
    for ind, bio_name in enumerate(data_columns[11:]):
        name_dict[bio_name] = f'epi{ind+1}'
    full_df.rename(columns=name_dict, inplace=True)

    # For the test file we use only reads with threshold >= 100
    wt_cond = full_df.wt_reads_sum >= 100
    esp_cond = full_df.esp_reads_sum >= 100
    hf_cond = full_df.hf_reads_sum >= 100

    # The test fi;e need to be the same for all models
    test_df = full_df[wt_cond & esp_cond & hf_cond].sample(frac=0.15).sort_index()
    train_val_full_df = full_df.drop(test_df.index)
    train_val_55K_df = full_df[wt_cond | esp_cond | hf_cond].drop(test_df.index)

    # Full data (No threshold)
    dir_path = 'data/pre_train/DeepHF_full/'
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    valid_full_df = train_val_full_df.sample(frac=0.1).sort_index()
    train_full_df = train_val_full_df.drop(valid_full_df.index)

    test_df.to_csv(dir_path + 'test.csv', index=False)
    train_val_full_df.to_csv(dir_path + 'train_valid.csv', index=False)
    valid_full_df.to_csv(dir_path + 'valid.csv', index=False)
    train_full_df.to_csv(dir_path + 'train.csv', index=False)

    # 55k data (100 reads threshold)
    dir_path = 'data/pre_train/DeepHF_55k/'
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    valid_55k_df = train_val_55K_df.sample(frac=0.1).sort_index()
    train_55k_df = train_val_55K_df.drop(valid_55k_df.index)

    test_df.to_csv(dir_path + 'test.csv', index=False)
    train_val_55K_df.to_csv(dir_path + 'train_valid.csv', index=False)
    valid_55k_df.to_csv(dir_path + 'valid.csv', index=False)
    train_55k_df.to_csv(dir_path + 'train.csv', index=False)


def prepare_sequences(reads_sum, dir_path, old):
    if os.path.exists(dir_path + 'train_seq.pkl'):
        print('Sequence files already prepared -> returning')
        return
    dataframes = ['test', 'valid', 'train']
    enzymes = ['wt', 'esp', 'hf']

    for df_name in dataframes:
        df_path = dir_path + df_name + '.csv'
        print(f'Preparing {df_path}')
        df = pd.read_csv(df_path)
        sequence = MultiSeq()
        for index, row in df.iterrows():
            print('line: {}'.format(index))
            seq = row['21mer']

            biofeat = []
            for i in range(1, 12):
                biofeat.append(row['epi{}'.format(i)])

            for enzyme in enzymes:
                enzyme_eff = row[f'{enzyme}_mean_eff']

                if old:
                    if not math.isnan(enzyme_eff):
                        sequence.add_seq(enzyme, seq, biofeat, enzyme_eff)

                else:
                    enzyme_reads = row[f'{enzyme}_reads_sum']
                    if reads_sum:
                        if enzyme_reads >= 1:
                            sequence.add_seq(enzyme, seq, biofeat, enzyme_eff, enzyme_reads)
                    else:
                        if enzyme_reads >= 100:
                            sequence.add_seq(enzyme, seq, biofeat, enzyme_eff)
        with open(dir_path + f'{df_name}_seq.pkl', "wb") as fp:
            pickle.dump(sequence, fp)



# Data structure objects
class MultiSeq(object):
    # This class is the serialized data for all 3 cas9 types
    def __init__(self):
        self.enzymes_seq = {'wt': Seq(), 'esp': Seq(), 'hf': Seq()}

    def add_seq(self, enzyme, seq, biofeat, y, conf=None):
        self.enzymes_seq[enzyme].add_seq(seq, biofeat, y, conf)


class Seq(object):
    # This class is the serialized data for one enzyme only
    def __init__(self):
        self.X = np.empty((0, 22), np.uint8)
        self.X_biofeat = np.empty((0, 11), np.float16)
        self.y = np.empty(0, np.float16)
        self.confidence = np.empty(0, np.uint16)


    def add_seq(self, seq, biofeat, y, conf=None):
        # The '0' represent the beginning of the sequence (help for the RNN)
        mer_array = np.array([0], dtype=np.uint8)

        for char in seq:
            num = char2int(char)
            mer_array = np.append(mer_array, np.array([num], dtype=np.uint8), axis=0)

        mer_array = np.expand_dims(mer_array, 0)
        self.X = np.concatenate((self.X, mer_array), axis=0)

        biofeat = np.array([float(epi) for epi in biofeat], dtype=np.float16)
        biofeat = np.expand_dims(biofeat, 0)
        self.X_biofeat = np.concatenate((self.X_biofeat,biofeat), axis=0)
        self.y = np.append(self.y, np.array([y], dtype=np.float16), axis=0)

        if conf is not None:
            self.confidence = np.append(self.confidence, np.array([conf], dtype=np.uint16), axis=0)


    # def get_data(self):
    #     if self.has_biofeat:
    #         return self.X, self.X_biofeat, self.y
    #     else:
    #         return self.X, self.y