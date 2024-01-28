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

# def get_crispor():
#     def get_PAM(row):
#
#         gRNA_seq = row['gRNA']
#         full_seq = row['full_seq'][123: 154].upper()
#
#         assert gRNA_seq == full_seq[5:-6]
#         assert full_seq[-5:-3] == 'GG'
#
#         return full_seq
#
#
#
#     def get_avg_indel(row):
#         assert row['gRNA_x'] == row['gRNA_y']
#         mean_eff = (row['total_indel_eff_x'] + row['total_indel_eff_y']) / 2
#         return mean_eff
#
#     if os.path.exists('data/edited_dataframes/crispor_full.csv'):
#         crispor_df = pd.read_csv('data/edited_dataframes/crispor_full.csv')
#         return crispor_df
#
#     indel_D10_df = pd.read_excel('data/main_dataframes/Crispor.xlsx', sheet_name='spCas9_eff_D10-dox')
#     indel_D10_df = indel_D10_df[['Surrogate ID', 'gRNA', 'total_indel_eff']]
#
#     indel_D8_df = pd.read_excel('data/main_dataframes/Crispor.xlsx', sheet_name='spCas9_eff_D8-dox')
#     indel_D8_df = indel_D8_df[['Surrogate ID', 'gRNA', 'total_indel_eff']]
#
#     avg_indel_df = pd.merge(indel_D10_df, indel_D8_df, on='Surrogate ID', how="left")
#     avg_indel_df.dropna(inplace=True)
#     avg_indel_df['mean_eff'] = avg_indel_df.apply(lambda x: get_avg_indel(x), axis=1)
#     avg_indel_df = avg_indel_df[['Surrogate ID', 'gRNA_x', 'mean_eff']]
#     avg_indel_df.rename(columns={'gRNA_x': 'gRNA'}, inplace=True)
#
#
#     pam_df = pd.read_excel('data/main_dataframes/Crispor.xlsx', sheet_name='TRAP12K microarray oligos')
#     pam_df = pam_df[['Surrogate ID', 'sequences (5\'to3\')']]
#     pam_df.rename(columns={'sequences (5\'to3\')': 'full_seq'}, inplace=True)
#
#
#     new_df = pd.merge(avg_indel_df, pam_df, on='Surrogate ID', how="left")
#
#     new_df['gRNA'] = new_df.apply(lambda x: get_PAM(x), axis=1)
#     new_df.drop('full_seq', axis='columns', inplace=True)
#     new_df.to_csv('data/edited_dataframes/crispor_full.csv', index=False)
#
#     return new_df
# def get_kim():
#     def get_seq(row):
#         seq = row['Target context sequence (4+20+3+3)'][4:25]
#         return seq
#
#     if os.path.exists('data/edited_dataframes/kim_full.csv'):
#         kim_df = pd.read_csv('data/edited_dataframes/kim_full.csv')
#         return kim_df
#
#     kim_train_df = pd.read_excel('data/main_dataframes/kim.xlsx', sheet_name='HT_Cas9_Train')
#     kim_test_df = pd.read_excel('data/main_dataframes/kim.xlsx', sheet_name='HT_Cas9_Test')
#
#     kim_train_df['gRNA'] = kim_train_df.apply(lambda x: get_seq(x), axis=1)
#     kim_train_df = kim_train_df[['gRNA', 'Background subtracted indel (%)']]
#     kim_train_df.rename(columns={'Background subtracted indel (%)': 'mean_eff'}, inplace=True)
#
#
#     kim_test_df['gRNA'] = kim_test_df.apply(lambda x: get_seq(x), axis=1)
#     kim_test_df = kim_test_df[['gRNA', 'Background subtracted indel frequencies\n(average, %)']]
#     kim_test_df.rename(columns={'Background subtracted indel frequencies\n(average, %)': 'mean_eff'}, inplace=True)
#
#     kim_df = pd.concat([kim_train_df, kim_test_df])
#     kim_df.to_csv('data/edited_dataframes/kim_full.csv', index=False)
#     return kim_df
# def linear_reg(crispor_df, kim_df):
#     # Find the identical gRNAs in both df
#     merged_df = pd.merge(crispor_df, kim_df, on='gRNA', how="inner")
#     merged_df.rename(columns={'mean_eff_x': 'crispron_eff', 'mean_eff_y': 'kim_eff'}, inplace=True)
#
#     # Train linear regresiion from kim to crispron
#     X_train = merged_df.kim_eff
#     y_train = merged_df.crispron_eff
#
#     LR = LinearRegression()
#     LR.fit(X_train.values.reshape(-1,1), y_train.values)
#
#     # Transform the full kim data
#     predictions = LR.predict(kim_df.mean_eff.values.reshape(-1,1))
#     kim_df['mean_eff_scaled'] = predictions
#
#     # Concantenate both dataframes
#     concat_df = pd.concat([crispor_df, kim_df])
#     return concat_df
# def get_data():
#     crispor_df = get_crispor()
#     kim_df = get_kim()
#
#     concat_df = linear_reg(crispor_df, kim_df)
#     return concat_df
# def get_model():
#     a=0



# full_df = get_data()
# model = get_model()

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

        main_file = open('data/main_dataframes/U6T7.tsv')

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

        df['downstream'] = df['longSeq100Bp'].map(lambda x: x[25:30])
        df['upstream'] = df['longSeq100Bp'].map(lambda x: x[53:55])
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
        for dataset in datasets:
            dataset_df = pd.read_csv(dir_path + f'set{set}/{dataset}.csv')
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


