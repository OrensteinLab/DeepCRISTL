import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import CRISPRspec_CRISPRoff_pipeline as pipe
import pickle


def prepare_inputs(config):
    if check_files_already_prepared(config):
        print('Files are allready prepared')
    else:
        prepare_files(config)
    if check_sequences_prepared(config):
        print('Sequences are allready prepared')
    else:
        prepare_sequences(config)


def prepare_files(config):

    csv_path = 'tool data/datasets/' + config.new_data_path + '.csv'

    df = pd.read_csv(csv_path)
    df['23mer'] = df['30mer'].map(lambda x: x[4:27])


    pipe.read_energy_parameters('data/model/energy_dics.pkl')
    global RNAFOLD_EXE
    RNAFOLD_EXE = 'RNAfold'

    df["CRISPRoff_score"] = df.apply(lambda x:
                                        pipe.get_eng(x['23mer'], x['23mer'], pipe.calcRNADNAenergy, GU_allowed=False,
                                                pos_weight=True, pam_corr=True, grna_folding=True, dna_opening=True,
                                                dna_pos_wgh=False), axis=1)
    
    # create folder if it does not exist and save the preprocessed file
    if not os.path.exists('tool data/datasets/' + config.new_data_path):
        os.makedirs('tool data/datasets/' + config.new_data_path)
    df.to_csv('tool data/datasets/' + config.new_data_path  + '/preprocessed.csv', index=False)


def check_files_already_prepared(config):
    if os.path.exists('tool data/datasets/' + config.new_data_path + '/preprocessed.csv'):
        return True
    else:
        return False

def check_sequences_prepared(config):
    path = 'tool data/datasets/' + config.new_data_path + '/train_val_seq.pkl'
    if os.path.exists(path):
        return True
    else:
        return False


def prepare_sequences(config):

    path = 'tool data/datasets/' + config.new_data_path + f'/'
    if not os.path.exists(path):
        os.makedirs(path)

    df = pd.read_csv('tool data/datasets/' + config.new_data_path + '/preprocessed.csv')
    # save the preprocessed file
    df.to_csv(path + 'train_val.csv', index=False)


    sequence = Seq()
    for index, row in df.iterrows():
        print('line: {}'.format(index))
        seq = row['30mer']
        dg = row['CRISPRoff_score']
        eff = row['mean_eff']
        sequence.add_seq(seq, dg, eff)

        with open(path + f'train_val_seq.pkl', "wb") as fp:
            pickle.dump(sequence, fp)


def onehot(x):
    z = list()
    for y in list(x):
        if y in "Aa":  z.append(0)
        elif y in "Cc": z.append(1)
        elif y in "Gg": z.append(2)
        elif y in "TtUu": z.append(3)
        else:
            print("Non-ATGCU character1 Q2A3SWE41Q")
            raise Exception
    return z

# Data structure objects
class Seq(object):
    # This class is the serialized data for one enzyme only
    def __init__(self, downstream_size=24, upstream_size=13):
        self.X = np.empty((0, 30, 4), np.uint8)
        self.dg = np.empty(0, np.float16)
        self.y = np.empty(0, np.float16)


    def add_seq(self, seq, dg, y):
        mer_array = np.expand_dims(np.eye(4)[onehot(seq)], axis=0)
        self.X = np.concatenate((self.X, mer_array), axis=0)

        self.dg = np.append(self.dg, np.array([dg], dtype=np.float16), axis=0)
        self.y = np.append(self.y, np.array([y], dtype=np.float16), axis=0)


