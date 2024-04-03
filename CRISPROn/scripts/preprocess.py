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
    dir_path = f'data/tl_train/{config.tl_data_category}/{config.tl_data}/'

    if config.tl_data_category == 'U6T7':
        prepare_u6_t7_files(config, dir_path)

    if config.tl_data_category == 'leenay':
        prepare_leenay_files(config, dir_path)

    if config.tl_data_category == 'crispr_il':
        prepare_crispr_il(config, dir_path)



    prepare_sequences(dir_path=dir_path)

def create_fasta(row, fasta_23, fasta_30):
    fasta_23.write(f'>{row.guide}\n')
    fasta_23.write(row['23mer'] + '\n')
    fasta_30.write(f'>{row.guide}\n')
    fasta_30.write(row['downstream'] + row['23mer'] + row['upstream'] + '\n')
    a=0


def prepare_u6_t7_files(config, dir_path):

    main_dataframes_path = 'data/main_dataframes/U6T7/'
    if not os.path.exists(main_dataframes_path):
        os.makedirs(main_dataframes_path)

        main_file = open('data/main_dataframes/target.tsv')

        headers = main_file.readline()
        names = []
        for line in main_file.readlines():
            dataset_name = line.split('\t')[0]
            if dataset_name not in names:
                dataset_file = open(f'{main_dataframes_path}{dataset_name}.tsv', 'w')
                names.append(dataset_name)
                dataset_file.write(headers)
            dataset_file.write(line)

    if os.path.exists(dir_path + 'set4/test_crispr_on.csv'):
        return
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if os.path.exists(dir_path + 'full_df.csv'):
        df = pd.read_csv(dir_path + 'full_df.csv')
    else:
        datasets = ['train', 'valid', 'test']


        df = pd.read_csv(f'data/main_dataframes/U6T7/{config.tl_data}.tsv', sep='\t')
        # df.rename(columns={'seq': '21mer'}, inplace=True)
        # df['21mer'] = df['21mer'].map(lambda x: x[0:21])

        df['23mer'] = df['longSeq100Bp'].map(lambda x: x[30:53])
        longSeq100Bp = df.longSeq100Bp.values
        mers = df['23mer'].values
        seqs = df.seq.values

        for ind, (mer, long, seq) in enumerate(zip(mers, longSeq100Bp, seqs)):
            if mer[:20] != seq:
                print(f'Ind: {ind}, mer: {mer}, seq: {seq}, long: {long}')
        # assert df['23mer'].equals(df['seq'])

        df['downstream'] = df['longSeq100Bp'].map(lambda x: x[26:30])
        df['upstream'] = df['longSeq100Bp'].map(lambda x: x[53:56])
        df['30mer'] = df['downstream'] + df['23mer'] + df['upstream']

        mean_eff_col_name = 'modFreq'
        df.rename(columns={mean_eff_col_name: 'mean_eff'}, inplace=True)
        scaler = MinMaxScaler()
        df.mean_eff = scaler.fit_transform(df.mean_eff.to_numpy().reshape(-1, 1)).reshape(-1)

        # fasta_23 = open(dir_path + '23mer.fa', 'w')
        # fasta_30 = open(dir_path + '30mer.fa', 'w')
        #
        # df.apply(lambda x: create_fasta(x, fasta_23, fasta_30), axis=1)

        pipe.read_energy_parameters('data/model/energy_dics.pkl')
        global RNAFOLD_EXE
        RNAFOLD_EXE = 'RNAfold'

        df["CRISPRoff_score"] = df.apply(lambda x:
                                         pipe.get_eng(x['23mer'], x['23mer'], pipe.calcRNADNAenergy, GU_allowed=False,
                                                 pos_weight=True, pam_corr=True, grna_folding=True, dna_opening=True,
                                                 dna_pos_wgh=False), axis=1)
        df.to_csv(dir_path + 'full_df.csv', index=False)

    eng_df = df[['guide', '30mer', 'CRISPRoff_score']]
    datasets = ['train_valid', 'train', 'valid', 'test']
    for set in range(5):
        if not os.path.exists(dir_path + f'set{set}/'):
            os.mkdir(dir_path + f'set{set}/')
        for dataset in datasets:
            dataset_df = pd.read_csv('../' + dir_path + f'set{set}/{dataset}.csv')
            merged = pd.merge(dataset_df, eng_df, how='inner', on=['guide'])
            merged.to_csv(dir_path + f'set{set}/{dataset}_crispr_on.csv', index=False)

def prepare_leenay_files(config, dir_path):

    if os.path.exists(dir_path + 'set4/test_crispr_on.csv'):
        return
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if os.path.exists(dir_path + 'full_df.csv'):
        df = pd.read_csv(dir_path + 'full_df.csv')
    else:
        datasets = ['train', 'valid', 'test']


        df = pd.read_csv(f'data/main_dataframes/leenay_full_data.csv')

        df['23mer'] = df['21mer'].map(lambda x: x + 'GG')
        # longSeq100Bp = df.longSeq100Bp.values
        # mers = df['23mer'].values
        # seqs = df.seq.values
        #
        # for ind, (mer, long, seq) in enumerate(zip(mers, longSeq100Bp, seqs)):
        #     if mer[:20] != seq:
        #         print(f'Ind: {ind}, mer: {mer}, seq: {seq}, long: {long}')
        # # assert df['23mer'].equals(df['seq'])

        df['30mer'] = df['downstream'].map(lambda x: x[-4:]) + df['23mer'] + df['upstream'].map(lambda x: x[:3])

        # scaler = MinMaxScaler()

        # fasta_23 = open(dir_path + '23mer.fa', 'w')
        # fasta_30 = open(dir_path + '30mer.fa', 'w')
        #
        # df.apply(lambda x: create_fasta(x, fasta_23, fasta_30), axis=1)

        pipe.read_energy_parameters('data/model/energy_dics.pkl')
        global RNAFOLD_EXE
        RNAFOLD_EXE = 'RNAfold'

        df["CRISPRoff_score"] = df.apply(lambda x:
                                         pipe.get_eng(x['23mer'], x['23mer'], pipe.calcRNADNAenergy, GU_allowed=False,
                                                 pos_weight=True, pam_corr=True, grna_folding=True, dna_opening=True,
                                                 dna_pos_wgh=False), axis=1)
        df.to_csv(dir_path + 'full_df.csv', index=False)

    eng_df = df[['name', '30mer', 'CRISPRoff_score']]
    datasets = ['train_valid', 'train', 'valid', 'test']
    for set in range(5):
        if not os.path.exists(dir_path + f'set{set}/'):
            os.mkdir(dir_path + f'set{set}/')
        for dataset in datasets:
            dataset_df = pd.read_csv('../' + dir_path + f'set{set}/{dataset}.csv')
            merged = pd.merge(dataset_df, eng_df, how='inner', on=['name'])
            merged.to_csv(dir_path + f'set{set}/{dataset}_crispr_on.csv', index=False)

def prepare_crispr_il(config, dir_path):
    if os.path.exists(dir_path + 'set4/test_crispr_on.csv'):
        return
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if os.path.exists(dir_path + 'full_df.csv'):
        df = pd.read_csv(dir_path + 'full_df.csv')
    else:
        datasets = ['train', 'valid', 'test']


        df = pd.read_csv(f'data/main_dataframes/crispr_il/{config.tl_data}.tsv', sep='\t')
        df.rename(columns={'Unnamed: 0': 'guide'}, inplace=True)


        seq = df['sgRNA_seq_0_char']
        for i in range(1, 23):
            col = f'sgRNA_seq_{i}_char'
            seq = seq + df[col]
        df['23mer'] = seq
        df['21mer'] = df['23mer'].map(lambda x: x[:21])

        up_seq = df['upstream_seq_16_char']
        for i in range(17, 20):
            col = f'upstream_seq_{i}_char'
            up_seq = up_seq + df[col]
        df['upstream'] = up_seq

        down_seq = df['downstream_seq_0_char']
        for i in range(1, 3):
            col = f'downstream_seq_{i}_char'
            down_seq = down_seq + df[col]
        df['downstream'] = down_seq


        df['30mer'] = df['downstream'] + df['23mer'] + df['upstream']

        mean_eff_col_name = 'efficiency'
        df.rename(columns={mean_eff_col_name: 'mean_eff'}, inplace=True)

        pipe.read_energy_parameters('data/model/energy_dics.pkl')
        global RNAFOLD_EXE
        RNAFOLD_EXE = 'RNAfold'

        df["CRISPRoff_score"] = df.apply(lambda x:
                                         pipe.get_eng(x['23mer'], x['23mer'], pipe.calcRNADNAenergy, GU_allowed=False,
                                                 pos_weight=True, pam_corr=True, grna_folding=True, dna_opening=True,
                                                 dna_pos_wgh=False), axis=1)
        df.to_csv(dir_path + 'full_df.csv', index=False)

    eng_df = df[['21mer', '30mer', 'CRISPRoff_score']]
    datasets = ['train_valid', 'train', 'valid', 'test']
    for set in range(5):
        for dataset in datasets:
            dataset_df = pd.read_csv(dir_path + f'set{set}/{dataset}.csv')
            merged = pd.merge(dataset_df, eng_df, how='inner', on=['21mer'])
            merged.to_csv(dir_path + f'set{set}/{dataset}_crispr_on.csv', index=False)

    a=0


def prepare_sequences(dir_path):
    if os.path.exists(dir_path + 'set4/train_seq.pkl'):
        print('Sequence files already prepared -> returning')
        return
    dataframes = ['test', 'valid', 'train']

    for set in range(5):
        set_path = dir_path + f'set{set}/'
        for df_name in dataframes:
            df_path = set_path + df_name + '_crispr_on.csv'
            print(f'Preparing {df_path}')
            df = pd.read_csv(df_path)

            sequence = Seq()
            for index, row in df.iterrows():
                print('line: {}'.format(index))
                seq = row['30mer']
                dg = row['CRISPRoff_score']
                eff = row['mean_eff']
                sequence.add_seq(seq, dg, eff)


            with open(set_path + f'{df_name}_seq.pkl', "wb") as fp:
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


