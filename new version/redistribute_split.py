

import numpy as np
import matplotlib.pyplot as plt

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

def check_stats(hamming_matrix, sets):
    print('Checking stats...')
    neighbores_per_row = np.sum(hamming_matrix, axis=1) 

    plt.hist(neighbores_per_row, bins=100)
    plt.title('Neighbor distribution, row-legth: {}'.format(len(hamming_matrix)))
    plt.show()

    # Show the size of the sets using a histogram
    set_sizes = [len(s) for s in sets]
    plt.hist(set_sizes, bins=20)
    plt.title('Set size distribution, number of sets: {}'.format(len(sets)))
    plt.show()



def get_sets(neighborehood_matrix, combined_df):
    length = len(neighborehood_matrix)
    sets = []
    used_indexes = []

    for i in range(length):
        if i in used_indexes:
            continue
        else:
            sequence_set = get_bfs(neighborehood_matrix, i)
            sets.append(sequence_set)
            used_indexes.extend(sequence_set)


def get_bfs(neighborehood_matrix, start_index):
    # Returns a set of all sequences that are connected to the start_index in the neighborehood_matrix
    # Uses a BFS algorithm
    queue = []
    queue.append(start_index)
    visited = []
    while len(queue) > 0:
        current_index = queue.pop(0)
        visited.append(current_index)
        for i in range(len(neighborehood_matrix[current_index])):
            if neighborehood_matrix[current_index][i] == 1 and i not in visited:
                queue.append(i)

    return visited

def save_hamming_matrix(hamming_matrix):
    print('Saving hamming matrix...')
    np.save('hamming_matrix', hamming_matrix)

def load_hamming_matrix():
    print('Loading hamming matrix...')
    return np.load('hamming_matrix.npy')

def check_hamming_matrix_exists():
    return os.path.isfile('hamming_matrix.npy')

def main():
    hamming_matrix_is_already_calculated = check_hamming_matrix_exists()
    if not hamming_matrix_is_already_calculated:
        combined_df = combine_data_files()
        hamming_matrix = calculate_hamming_distance_matix(combined_df)
        save_hamming_matrix(hamming_matrix)
    hamming_matrix = load_hamming_matrix()
    neighborehood_matrix = apply_neighbore_filter(hamming_matrix, max_distance=4)
    sets = get_sets(neighborehood_matrix, combined_df)

    check_stats(neighborehood_matrix, sets)

if __name__ == '__main__':
    main()