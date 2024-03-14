

import numpy as np
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

import os
import sys
import pandas as pd
import Levenshtein as lev

MAX_DISTANCE = 4
TL_BASE_PATH = '../data/tl_train/U6T7/'

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

def combine_data_files(remove_tl_leakage=False):
    print('Combining data files...')



    files = ['test.csv', 'train.csv', 'valid.csv']
    df = pd.DataFrame()
    for file in files:
        df = df._append(pd.read_csv(file))

    # Remove rows in df that are distance 4 or less from tl_dataframes in the field '21mer' usign hammings distance
    if remove_tl_leakage:
        print('Removing TL leakage...')
        new_df = pd.DataFrame()
        tl_folders, tl_names = get_folders(TL_BASE_PATH)
        tl_folders = [get_first_set(TL_BASE_PATH + folder) for folder in tl_folders]
        tl_dataframes = [get_df_from_folder(folder) for folder in tl_folders]
        removed = 0
        for sequence in df['21mer'].values:
            if not any(lev.distance(sequence, tl_sequence) <= MAX_DISTANCE for tl_df in tl_dataframes for tl_sequence in tl_df['21mer'].values):
                new_df = new_df._append(df[df['21mer'] == sequence])
            else:
                removed += 1

        df = new_df
        print('Removed {} rows'.format(removed))



    # sort by index
    df = df.sort_values(by=['21mer'])



    return df

def calculate_hamming_distance_matix(df):
    print('Calculating hamming distance matrix...')
    sequences = df['21mer'].values
    hamming_matrix = np.zeros((len(sequences), len(sequences)))
    done = 0
    total = len(sequences)
    for i in range(len(sequences)):
        for j in range(len(sequences)):
            hamming_matrix[i][j] = lev.distance(sequences[i], sequences[j])
        done += 1
        if done % 100 == 0:
              print('Done: {}%'.format(round((done / total) * 100, 1))) 

    return hamming_matrix


def apply_neighbor_filter(hamming_matrix, max_distance):
    print('Applying neighbor filter...')
    hamming_matrix[hamming_matrix <= max_distance] = 1
    hamming_matrix[hamming_matrix > max_distance] = 0

    return hamming_matrix

def check_stats(neighborehood_matrix, sets):
    print('Checking stats...')
    neighbores_per_row = np.sum(neighborehood_matrix, axis=1) 

    plt.hist(neighbores_per_row, bins=100)
    plt.title('Neighbor distribution, row-legth: {}'.format(len(neighborehood_matrix)))
    plt.savefig('neighbor_distribution.png')

    # clear
    plt.clf()


    # Show the size of the sets using a histogram
    set_sizes = [len(s) for s in sets]
    plt.hist(set_sizes, bins=20)
    plt.title('Set size distribution, number of sets: {}'.format(len(sets)))
    plt.savefig('set_size_distribution.png')



def get_sets(neighborehood_matrix):
    print('Getting sets...')
    length = len(neighborehood_matrix)
    sets = []
    used_indexes = []

    for i in range(length):
        if i in used_indexes:
            continue
        else:
            print('Getting set for index: {}/{} ({:.2f}%)'.format(i, length, (i / length) * 100))
            sequence_set = get_bfs(neighborehood_matrix, i)
            sets.append(sequence_set)
            used_indexes.extend(sequence_set)

    return sets

def save_sets (sets):
    print('Saving sets...')
    # save with pickle
    with open('sets.pickle', 'wb') as f:
        pickle.dump(sets, f)



def load_sets():
    print('Loading sets...')
    with open('sets.pickle', 'rb') as f:
        sets = pickle.load(f)

    return sets


def check_sets_missing():
    return not os.path.isfile('sets.pickle')
  

def get_bfs(neighborehood_matrix, start_index):
    # Returns a set of all sequences that are connected to the start_index in the neighborehood_matrix
    length = len(neighborehood_matrix)
    queue = []
    queue.append(start_index)
    visited = set()

    while queue:
        current_index = queue.pop(0)
        visited.add(current_index)

        for i in range(length):
            if neighborehood_matrix[current_index][i] == True and i not in visited:
                queue.append(i)

    return visited

def save_hamming_matrix(hamming_matrix):
    print('Saving hamming matrix...')
    np.save('hamming_matrix', hamming_matrix)

def load_hamming_matrix():
    print('Loading hamming matrix...')
    return np.load('hamming_matrix.npy')

def check_hamming_matrix_missing():
    return not os.path.isfile('hamming_matrix.npy')

def check_neighborehood_matrix_missing():
    return not os.path.isfile('neighborehood_matrix.npy')

def save_neighborehood_matrix(neighborehood_matrix):
    neighborehood_matrix = neighborehood_matrix.astype(bool)
    print('Saving neighborehood matrix...')
    np.save('neighborehood_matrix', neighborehood_matrix)

def load_neighborehood_matrix():
    print('Loading neighborehood matrix...')
    return np.load('neighborehood_matrix.npy')




def train_test_split(df, sets, test_ratio):

    # order sets by size
    sets = sorted(sets, key=len)

    train_ratio = 1 - test_ratio

    train = pd.DataFrame()
    test = pd.DataFrame()

    for i, sequence_set in enumerate(sets):

        normalized_train_size = len(train) * test_ratio
        normalized_test_size = len(test) * train_ratio

        if normalized_train_size < normalized_test_size:
            train = train._append(df.iloc[list(sequence_set)])
        else:
            test = test._append(df.iloc[list(sequence_set)])

    return train, test

def random_train_test_split(df, sets, test_ratio, seed):
    # Order sets by size
    sets = sorted(sets, key=len)

    np.random.seed(seed)

    np.random.shuffle(sets)

    train_ratio = 1 - test_ratio

    train = pd.DataFrame()
    test = pd.DataFrame()      

    for i, sequence_set in enumerate(sets):
        normalized_train_size = len(train) * test_ratio
        normalized_test_size = len(test) * train_ratio

        if normalized_train_size < normalized_test_size:
            train = train.append(df.iloc[list(sequence_set)])
        else:
            test = test.append(df.iloc[list(sequence_set)])

    return train, test  


def train_val_split(df, val_ratio):
    val = df.sample(frac=val_ratio)
    train = df.drop(val.index)

    return train, val

def save_files(train, val, test, remove_tl_leakage=False):

    # Order by index
    train = train.sort_values(by=['21mer'])
    val = val.sort_values(by=['21mer'])
    if not remove_tl_leakage:
        test = test.sort_values(by=['21mer'])


    # create folder
    if not os.path.exists('output'):
        os.makedirs('output')
    
    # save files
    train.to_csv('output/train.csv', index=False)
    val.to_csv('output/valid.csv', index=False)
    test.to_csv('output/test.csv', index=False)

def remove_incomplete_rows(df):
    df = df.dropna()

    return df

# Takes a dataframe and redistributes it to train, val and test for a TL dataset
def redistribute_tl_data(df, seed):
    hamming_matrix = calculate_hamming_distance_matix(df)
    neighborehood_matrix = apply_neighbor_filter(hamming_matrix, max_distance=MAX_DISTANCE)
    sets = get_sets(neighborehood_matrix)
    train, test = random_train_test_split(df, sets, test_ratio=0.2, seed=seed)
    train, val = train_val_split(train, val_ratio=0.2)

    return train, val, test



def main():
    # check args length
    if len(sys.argv)==2 and sys.argv[1] == 'remove_tl_leakage':
        REMOVE_TL_LEAKAGE = True
    else:
        REMOVE_TL_LEAKAGE = False

    if REMOVE_TL_LEAKAGE:
        train = combine_data_files(remove_tl_leakage=True)
        test = pd.DataFrame(columns=train.columns)
        train, val = train_val_split(train, val_ratio=0.1)
        save_files(train, val, test, remove_tl_leakage=True)
    else:
        # if check_neighborehood_matrix_missing():
        #     if check_hamming_matrix_missing():
        #         combined_df = combine_data_files(REMOVE_TL_LEAKAGE)
        #         hamming_matrix = calculate_hamming_distance_matix(combined_df)
        #         save_hamming_matrix(hamming_matrix)
        #     hamming_matrix = load_hamming_matrix()
        #     neighborehood_matrix = apply_neighbor_filter(hamming_matrix, max_distance=MAX_DISTANCE)
        #     save_neighborehood_matrix(neighborehood_matrix)

        # neighborehood_matrix = load_neighborehood_matrix()

        combined_df=  combine_data_files(REMOVE_TL_LEAKAGE)
        hamming_matrix = calculate_hamming_distance_matix(combined_df)
        neighborehood_matrix = apply_neighbor_filter(hamming_matrix, max_distance=MAX_DISTANCE)


        # print some of the neighborehood matrix
        if check_sets_missing():
            sets = get_sets(neighborehood_matrix)
            save_sets(sets)
        
        sets = load_sets()

        check_stats(neighborehood_matrix, sets)


        combined_df = combine_data_files()
        train, test = train_test_split(combined_df, sets, test_ratio=0.15)
        test = remove_incomplete_rows(test)
        train, val = train_val_split(train, val_ratio=0.1)

        save_files(train, val, test)



if __name__ == '__main__':
    main()