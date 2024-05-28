import pandas as pd
import os
import Levenshtein

PRETRAIN_GRNAS_PATH = '../data/main_dataframes/pretrain_grnas.csv'
LEENAY_PATH = '../data/main_dataframes/leenay_full_data.csv'
U6T7_PATH = '../data/main_dataframes/13059_2016_1012_MOESM14_ESM.tsv'


def get_pretrain_grnas(no_GG = False):
    pretrain_grnas = pd.read_csv(PRETRAIN_GRNAS_PATH)
    pretrain_grnas = set(pretrain_grnas['sequence'].values)

    if no_GG:
        # remove the last two characters from each sequence
        pretrain_grnas = set([seq[:-2] for seq in pretrain_grnas])
    
    return pretrain_grnas


def is_leaking(seq, grnas_set):
    for grna in grnas_set:
        distance = Levenshtein.hamming(seq, grna)
        if distance < 4:
            return True
    return False


def remove_leakage_leenay():
    leenay_df=  pd.read_csv(LEENAY_PATH)

    # print the number of sequences
    print('Number of sequences before removing leakage:', len(leenay_df))

    pretrain_grnas = get_pretrain_grnas(no_GG=True)
    leenay_df['is_leaking'] = leenay_df['21mer'].apply(lambda x: is_leaking(x, pretrain_grnas))
    leenay_df = leenay_df[leenay_df['is_leaking'] == False]
    # remove the is_leaking column
    leenay_df = leenay_df.drop(columns=['is_leaking'])

    leenay_df.to_csv(LEENAY_PATH, index=False)

    # print the number of sequences
    print('Number of sequences after removing leakage:', len(leenay_df))


def remove_leakage_U6T7():
    u6t7_df = pd.read_csv(U6T7_PATH, sep='\t')

    # print the number of sequences
    print('Number of sequences before removing leakage:', len(u6t7_df))

    pretrain_grnas = get_pretrain_grnas(no_GG=False)
    u6t7_df['is_leaking'] = u6t7_df['seq'].apply(lambda x: is_leaking(x, pretrain_grnas))
    u6t7_df = u6t7_df[u6t7_df['is_leaking'] == False]
    # remove the is_leaking column
    u6t7_df = u6t7_df.drop(columns=['is_leaking'])

    # save to tsv
    u6t7_df.to_csv(U6T7_PATH, sep='\t', index=False)

    # print the number of sequences
    print('Number of sequences after removing leakage:', len(u6t7_df))





if __name__ == '__main__':
    print ('Removing leakage from Leenay')
    remove_leakage_leenay()
    print ('Removing leakage from U6T7')
    remove_leakage_U6T7()
