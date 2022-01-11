from scipy import stats
import os
import pandas as pd
import numpy as np

def test_model(config, model, DataHandler, verbose=1):
    spearman_result = {}

    if verbose > 0:
        print('Testing model')

    test_input = [DataHandler['X_test'], DataHandler['X_biofeat_test']]
    if config.flanks:
        test_input += [DataHandler['up_test'], DataHandler['down_test']]

    if config.new_features:
        test_input += [DataHandler['new_features_test']]

    test_true_label = DataHandler['y_test']
    test_prediction = model.predict(test_input)
    spearman = stats.spearmanr(test_true_label, test_prediction)
    spearman_result[config.enzyme] = spearman[0]
    if verbose > 0:
        print(f'Spearman: {spearman}')

    return spearman_result

def save_results(config, enzyme, train_type, mean, spearmanr):
    results_path = f'results/transfer_learning/{config.tl_data}/set{config.set}/results.csv'
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path, index_col=0)
    else:
        rows = ['full_tl', 'full_tl_ensemble', 'LL_tl', 'LL_tl_ensemble', 'gl_tl', 'gl_tl_ensemble', 'no_em_tl', 'no_em_tl_ensemble', 'no_tl', 'no_tl_ensemble']
        results_df = pd.DataFrame(index=rows, columns=['wt', 'esp', 'hf', 'multi_task'])

    if results_df.shape[0] == 10:
        results_df.loc['no_pre_train'] = ['None', 'None', 'None', 'None']
        results_df.loc['no_pre_train_ensemble'] = ['None', 'None', 'None', 'None']
    if 'multi_task' not in results_df.columns:
        results_df['multi_task'] = np.nan
    results_df[enzyme][train_type] = '%.4f' % mean
    p_val = "{:e}".format(spearmanr[1])
    p_val = p_val[:5] + p_val[p_val.find('e'):]
    results_df[enzyme][train_type + '_ensemble'] = '%.4f ' % spearmanr[0] +  f' ({p_val})'

    results_df.to_csv(results_path, index=True)
