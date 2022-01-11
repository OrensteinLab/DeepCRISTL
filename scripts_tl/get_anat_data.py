import pandas as pd
import os
from scripts_tl import preprocess_tl

def get_coordinates(x):
    if x[0].split('_')[1] == 'PATCH':
        _, _, start, end, _ = x[0].split('_')
    else:
        _, start, end, _ = x[0].split('_')
    return int(start), int(end)

full_df = pd.read_csv('../data/main_dataframes/leenay_full_data.csv')

# Loading Anat features and dropping sequence columns
anat_test = pd.read_csv('../data/anat_data/ASSESS_regression_lightgbm_regression_0.testfeatures', sep='\t')
new_features_columns = [c for c in anat_test.columns if 'seq' not in c.lower()]
anat_test = anat_test[new_features_columns]
anat_test[['start', 'end']] = anat_test.apply(lambda x: pd.Series(get_coordinates(x)), axis=1)


anat_train = pd.read_csv('../data/anat_data/ASSESS_regression_lightgbm_regression_0.trainingfeatures', sep=',')
anat_train = anat_train[new_features_columns]
anat_train[['start', 'end']] = anat_train.apply(lambda x: pd.Series(get_coordinates(x)), axis=1)

test_indexes = []
for row in anat_test.iterrows():
    start, end = row[1]['start'], row[1]['end']
    index = int(full_df[(full_df['start'] == start) & (full_df['end'] == end)].index.values)
    test_indexes.append(index)

test_df = full_df.iloc[test_indexes, :]
train_val_df = full_df.drop(test_df.index)
valid_df = train_val_df.sample(frac=0.2, random_state=12).sort_index()
train_df = train_val_df.drop(valid_df.index)


test_df = pd.merge(test_df, anat_test, on='start')
test_df.drop(['Unnamed: 0', 'end_y'], axis='columns', inplace=True)

valid_df = pd.merge(valid_df, anat_train, on='start')
valid_df.drop(['Unnamed: 0', 'end_y'], axis='columns', inplace=True)

train_df = pd.merge(train_df, anat_train, on='start')
train_df.drop(['Unnamed: 0', 'end_y'], axis='columns', inplace=True)

dir_path = '../data/tl_train/leenay_anat/'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)


test_df.to_csv(dir_path + 'test.csv')
train_df.to_csv(dir_path + 'train.csv')
valid_df.to_csv(dir_path + 'valid.csv')

# pos_df = test_df[['start', 'end_x', 'pos']]
# pos_anat = anat_test[['Unnamed: 0', 'pos']

preprocess_tl.prepare_sequences(False, dir_path, False, add_new_features=True)



