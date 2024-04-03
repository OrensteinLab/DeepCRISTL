

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
            train = train.append(df.iloc[list(sequence_set)])
        else:
            test = test.append(df.iloc[list(sequence_set)])

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



def redistribute_dhf_pretrain_data(df):
    hamming_matrix = calculate_hamming_distance_matix(df)
    neighborehood_matrix = apply_neighbor_filter(hamming_matrix, max_distance=MAX_DISTANCE)
    sets = get_sets(neighborehood_matrix)
    train, test = train_test_split(df, sets, test_ratio=0.15)
    train, val = train_val_split(train, val_ratio=0.1)

    return train, val, test

