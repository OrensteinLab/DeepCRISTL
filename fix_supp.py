import pandas as pd
import os

BASE_PATH = 'data/main_dataframes/'
CORRECTION_FOLDER = 'data/main_dataframes/CrisprOR fixes/'


# EXPERIMENTS_TO_FIX = ['doench2016_hg19'
    
#     'hart2016-Rpe1Avg',
#                         'hart2016-Hct1162lib1Avg',
#                         'hart2016-HelaLib2Avg',
#                         'hart2016-HelaLib1Avg',
#                         ]




MAIN_FILE = '13059_2016_1012_MOESM14_ESM.tsv'


#CORRECTION_FILES = [CORRECTION_FOLDER + file for file in CORRECTION_FILES]
MAIN_FILE = BASE_PATH + MAIN_FILE



                    
def fix_row_length(file):
    # open the file and make sure each row has 21 columns and if not remove the last columns

    with open(file, 'r') as f:
        lines = f.readlines()
        lines = [line.strip().split('\t') for line in lines]
        lines = [line[:21] for line in lines]

    with open(file, 'w') as f:
        for line in lines:
            f.write('\t'.join(line) + '\n')

    return

    
def fix_ghandi():
    # open main file and fix the ghandi experiment
    with open(MAIN_FILE, 'r') as f:
        lines = f.readlines()
        lines = [line.strip().split('\t') for line in lines]
        for line in lines:
            if line[0] == 'gandhi2016_ci2':
                line[0] = 'ghandi2016_ci2'
    
    with open(MAIN_FILE, 'w') as f:
        for line in lines:
            f.write('\t'.join(line) + '\n')
    
    return

def get_experiments_to_fix():
    # open the main file
    bad_experiments = set()
    with open(MAIN_FILE, 'r') as f:
        lines = f.readlines()
        lines = [line.strip().split('\t') for line in lines]
        for line in lines[1:]:
            if len(line) != 21:
                bad_experiments.add(line[0])

    print(bad_experiments)
    return bad_experiments

def main():
    # fix the ghandi experiment name
    fix_ghandi() 

    # Get all the experiments that have a wrong number of columns
    bad_experiments = get_experiments_to_fix()
    correction_files = [''.join([CORRECTION_FOLDER,experiment, '.scores.tab']) for experiment in bad_experiments]
    # Check they all exist
    for file in correction_files:
        if not os.path.exists(file):
            print(f'File {file} does not exist')
    fix_row_length(MAIN_FILE)
    main_df = pd.read_csv(MAIN_FILE, sep='\t')
    fixes_dfs = [pd.read_csv(file, sep='\t') for file in correction_files]

    # for reach row in main_df, find the corresponding row in each fixes_df
    # and update the main_df row with the new values
    for i, row in main_df.iterrows():
        for df in fixes_dfs:
            # fix rows with matching 'dataset' and 'longSeq100Bp' values
            fix_row = df[(df['dataset'] == row['dataset']) & (df['longSeq100Bp'] == row['longSeq100Bp'])]
            if fix_row.shape[0] == 1:
                fix_values = fix_row.iloc[0]
                main_df.loc[i, fix_values.index] = fix_values
            
    main_df.to_csv(MAIN_FILE, sep='\t', index=False)

if __name__ == '__main__':
    main()