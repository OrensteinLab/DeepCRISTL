from scipy import stats
import pandas as pd
import os
import numpy as np

def test_model(config, model, DataHandler, verbose=1):
    spearman_result = {}

    if verbose > 0:
        print('Testing model')

    test_input = [DataHandler['X_test'], DataHandler['dg_test']]

    test_true_label = DataHandler['y_test']
    test_prediction = model.predict(test_input)
    spearman = stats.spearmanr(test_true_label, test_prediction)
    spearman_result = spearman[0]
    if verbose > 0:
        print(f'Spearman: {spearman}')

    return spearman_result


def save_results(config, set, train_type, mean, spearmanr):
    columns = ['full_tl', 'full_tl_ensemble', 'LL_tl', 'LL_tl_ensemble', 'gl_tl', 'gl_tl_ensemble', 'no_tl', 'no_tl_ensemble', 'no_pre_train', 'no_pre_train_ensemble', 'no_conv_tl', 'no_conv_tl_ensemble']
    results_path = f'results/transfer_learning/{config.tl_data}/results.csv'
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path, index_col=0)
        # add columns if they don't exist TODO: remove this
        for column in columns:
            if column not in results_df.columns:
                results_df[column] = [np.nan] * len(results_df)


    else:
        rows = []
        for i in range(5):
            rows.append(f'set{i}')
        results_df = pd.DataFrame(index=rows, columns=columns)

    results_df[train_type][f'set{set}'] = '%.4f' % mean
    p_val = "{:e}".format(spearmanr[1])
    p_val = p_val[:5] + p_val[p_val.find('e'):]
    results_df[train_type + '_ensemble'][f'set{set}'] = '%.4f ' % spearmanr[0] +  f' ({p_val})'

    results_df.to_csv(results_path, index=True)
