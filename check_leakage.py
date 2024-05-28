import pandas as pd
import os
import Levenshtein

BASE_PATH = 'data/tl_train/U6T7/'
PRE_TRAIN_PATH = 'data/pre_train/DeepHF_old/'

def get_folders(path):
    folders = []
    names = []
    for folder in os.listdir(path):
        if os.path.isdir(os.path.join(path, folder)):
            folders.append(folder)
            names.append(folder)
    return folders, names

def get_first_set(path):
    return path + '/set0/'


def get_df_from_folder(path):
    # open test.csv
    df = pd.read_csv(path + 'test.csv')
    # open train.csv
    df_train = pd.read_csv(path + 'train.csv')
    # open valid.csv
    df_valid = pd.read_csv(path + 'valid.csv')

    # concat all dataframes
    df = pd.concat([df, df_train, df_valid])
    return df


def get_pre_train_df():
    df = pd.read_csv(PRE_TRAIN_PATH + 'test.csv')
    df_train = pd.read_csv(PRE_TRAIN_PATH + 'train.csv')
    df_valid = pd.read_csv(PRE_TRAIN_PATH + 'valid.csv')

    df = pd.concat([df, df_train, df_valid])
    return df


def get_leakage(df, distance, pretrain_df):
    print('Calculating leakage for distance: ', distance)
    pretrain_sequences = pretrain_df['21mer'].values
    tl_sequences = df['21mer'].values

    leakage = 0

    for tl_seq in tl_sequences:
        # if hamming distance is less than distance
        if any(Levenshtein.hamming(tl_seq, pre_seq) <= distance for pre_seq in pretrain_sequences):
            leakage += 1

    return leakage


    
def make_stats_for_tl_datasets(dfs, names):
    # create a new dataframe
    df = pd.DataFrame()

    # add columns to the new dataframe
    df['dataset'] = names
    df['num_rows'] = [len(df) for df in dfs]

    pretrain_df = get_pre_train_df()


    df['leakage_3'] = [get_leakage(df,3, pretrain_df) for df in dfs]
    df['leakage_4'] = [get_leakage(df,4, pretrain_df) for df in dfs]

    # save 
    df.to_csv('tl_stats.csv', index=False)


def get_size_of_train_dataset_after_removal(dfs):
    pretrain_df = get_pre_train_df()
    # Check for each row in pretrain_df if it is in any of the tl datasets
    pretrain_sequences = pretrain_df['21mer'].values

    size_before = len(pretrain_df)
    to_remove_3 = 0
    to_remove_4 = 0

    for pretrain_sequence in pretrain_sequences:
        for df in dfs:
            for tl_sequence in df['21mer'].values:
                hamming = Levenshtein.hamming(pretrain_sequence, tl_sequence)
                if hamming <= 3:
                    to_remove_3 += 1
                    break


    for pretrain_sequence in pretrain_sequences:
        for df in dfs:
            for tl_sequence in df['21mer'].values:
                hamming = Levenshtein.hamming(pretrain_sequence, tl_sequence)
                if hamming <= 4:
                    to_remove_4 += 1
                    break


    print('Size before: ', size_before)
    print('Size after removal of 3: ', size_before - to_remove_3)
    print('Size after removal of 4: ', size_before - to_remove_4)

    # save it to a file
    with open('size_after_removal.txt', 'w') as f:
        f.write('Size before: ' + str(size_before) + '\n')
        f.write('Size after removal of 3: ' + str(size_before - to_remove_3) + '\n')
        f.write('Size after removal of 4: ' + str(size_before - to_remove_4) + '\n')



def main():
    folders, names = get_folders(BASE_PATH)
    folders = [get_first_set(BASE_PATH + folder) for folder in folders]
    dataframes = [get_df_from_folder(folder) for folder in folders]
    make_stats_for_tl_datasets(dataframes, names)

    get_size_of_train_dataset_after_removal(dataframes)

if __name__ == '__main__':
    main()