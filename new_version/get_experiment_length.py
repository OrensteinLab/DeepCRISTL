import pandas as pd


TL_BASE_PATH = '../data/main_dataframes/13059_2016_1012_MOESM14_ESM.tsv'

def main():
    # create csv file to write to
    file = open('experiment_lengths.csv', 'w')

    # write headers
    file.write('experiment_id, number_of_samples, train_samples, test_samples\n')


    df = pd.read_csv(TL_BASE_PATH, sep='\t')
    # get the unique experiment ids
    experiment_ids = df['dataset'].unique()

    # get the number of samples for each experiment
    for experiment_id in experiment_ids:
        number_of_samples = len(df[df['dataset'] == experiment_id])
        test_samples = 0.2 * number_of_samples
        train_samples = 0.8 * number_of_samples
        print('Experiment ID: ', experiment_id)
        print('Number of samples: ', number_of_samples)
        print('Train samples: ', train_samples)
        print('Test samples: ', test_samples)
        print('')
        file.write(f'{experiment_id},{number_of_samples},{train_samples},{test_samples}\n')

    file.close()


if __name__ == '__main__':
    main()
