

import numpy as np
import matplotlib.pyplot as plt
import pickle

import os
import pandas as pd
import Levenshtein as lev

# VAL is 10% of train
# Train test split is 85% 15%

def combine_data_files():
    print('Combining data files...')
    files = ['test.csv', 'train.csv', 'valid.csv']
    df = pd.DataFrame()
    for file in files:
        df = df._append(pd.read_csv(file))

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


def apply_neighbore_filter(hamming_matrix, max_distance):
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

def main():


    if check_neighborehood_matrix_missing():
        if check_hamming_matrix_missing():
            combined_df = combine_data_files()
            hamming_matrix = calculate_hamming_distance_matix(combined_df)
            save_hamming_matrix(hamming_matrix)
        hamming_matrix = load_hamming_matrix()
        neighborehood_matrix = apply_neighbore_filter(hamming_matrix, max_distance=4)
        save_neighborehood_matrix(neighborehood_matrix)

    neighborehood_matrix = load_neighborehood_matrix()
    # print some of the neighborehood matrix
    if check_sets_missing():
        sets = get_sets(neighborehood_matrix)
        save_sets(sets)
    
    sets = load_sets()
    check_stats(neighborehood_matrix, sets)

if __name__ == '__main__':
    main()