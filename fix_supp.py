import pandas as pd

BASE_PATH = 'data/main_dataframes/'
CORRECTION_FILES = ['hart2016-Rpe1Avg.scores.tab',
                    'hart2016-Hct1162lib1Avg.scores.tab',
                    'hart2016-HelaLib2Avg.scores.tab',
                    'hart2016-HelaLib1Avg.scores.tab',
                    'doench2016_hg19.scores.tab']

MAIN_FILE = '13059_2016_1012_MOESM14_ESM.tsv'


CORRECTION_FILES = [BASE_PATH + file for file in CORRECTION_FILES]
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

    




def main():
    fix_row_length(MAIN_FILE)
    main_df = pd.read_csv(MAIN_FILE, sep='\t')
    fixes_dfs = [pd.read_csv(file, sep='\t') for file in CORRECTION_FILES]

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