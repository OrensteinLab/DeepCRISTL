import pandas as pd
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# import shap
from keras.models import load_model
# import PyNonpar
# from PyNonpar import *
import scipy as sp
# import seaborn as sns
import time
from scripts_tl import data_handler_tl as dh_tl
from scripts_tl import training_util_tl
from scripts import models_util
from scripts_tl import ensemble_util_tl
import os
import numpy as np

expirements = ['xu2015TrainHl60', 'chari2015Train293T', 'hart2016-Rpe1Avg', 'hart2016-Hct1162lib1Avg',
               'hart2016-HelaLib1Avg', 'hart2016-HelaLib2Avg','xu2015TrainKbm7', 'doench2014-Hs' , 'doench2014-Mm',
               'doench2016_hg19', 'leenay']

T7_expirements = ['eschstruth', 'varshney2015', 'gagnon2014', 'shkumatavaPerrine', 'shkumatavaAngelo', 'shkumatavaOthers', 'teboulVivo_mm9', 'morenoMateos2015']
T7 = True

if T7:
    expirements = T7_expirements

crispr_il_expirements = ['WAL_M82_protoplasts_6.10.2021_with_features', 'HOP_U937_Human_Monocytes_6.10.2021_with_features']

rows_ensemble = ['gl_tl_ensemble','full_tl_ensemble', 'no_em_tl_ensemble','LL_tl_ensemble', 'no_tl_ensemble', 'no_pre_train_ensemble']
rows = ['gl_tl','full_tl', 'no_em_tl','LL_tl', 'no_tl', 'no_pre_train']
enzymes = ['multi_task']#['wt', 'esp', 'hf', 'multi_task']


def postprocess(config):
    # calc_avg_res(config)
    # compare_ensemble()
    # tl_veriations_hit_map()
    # tl_veriations_bar_plot()
    # tl_veriations_vs_crispron_bar_plot()
    final_table()
    # interpertate(config)
    # ensemble_models_curve(config)

def calc_avg_res(config):
    exps = expirements if config.tl_data_category == 'U6T7' else crispr_il_expirements

    for exp in exps:
        print(exp)
        sets_df_array = []
        train_types = rows + rows_ensemble
        avg_res_df = pd.DataFrame(index=train_types, columns=enzymes)
        error_df = avg_res_df.copy(deep=True)
        path = f'results/transfer_learning/{exp}/'
        for set in range(5):
            set_path = path + f'set{set}/results.csv'
            set_res_df = pd.read_csv(set_path, index_col=0)
            sets_df_array.append(set_res_df)

        for enzyme in enzymes:

            for train_type in train_types:
                if exp == 'leenay':
                    if enzyme in ['esp', 'hf']:
                        continue
                    if enzyme == 'wt' and train_type != 'LL_tl':
                        continue
                if train_type in ['no_pre_train', 'no_pre_train_ensemble'] and enzyme not in  ['wt', 'multi_task']:
                    continue
                val_arr = []
                for set in range(5):
                    val = sets_df_array[set][enzyme][train_type]

                    if (pd.isna(val) or 'nan' in val):
                        val = '0'
                    if '(' in val:
                        val = val.split(' ')[0]

                    val_arr.append(abs(float(val)))
                error_df[enzyme][train_type] = np.std(val_arr)
                avg_res_df[enzyme][train_type] = '%.4f' % (sum(val_arr)/len(val_arr))

        error_df.to_csv(path + 'std_results.csv')
        avg_res_df.to_csv(path + 'avg_results.csv')
    a=0


def compare_ensemble():
    single_model_res_arr = {}
    ensemble_model_res_arr = {}
    for exp in expirements:
        path = f'results/transfer_learning/{exp}/avg_results.csv'
        avg_res_df = pd.read_csv(path, index_col=0)
        single_model_res_arr[exp] = avg_res_df['wt']['LL_tl']
        ensemble_model_res_arr[exp] = avg_res_df['wt']['LL_tl_ensemble']


    # Wilcoxon rank sum test
    # p = PyNonpar.twosample.wilcoxon_mann_whitney_test(list(single_model_res_arr.values()), list(ensemble_model_res_arr.values()), alternative="less")
    p = sp.stats.wilcoxon(list(single_model_res_arr.values()), list(ensemble_model_res_arr.values()), zero_method='wilcox', correction=False, alternative='two-sided', mode='exact')
    # Plot
    plt.figure()
    # markers_arr = ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']
    marker_arr = [['x','b'], ['^', 'orange'], ['o', 'r'], ['o', 'b'], ['o', 'y'], ['o', 'purple'], ['x', 'g'],
                  ['<', 'g'], ['<', 'r'], ['<', 'orange']]#, ['s', 'brown']]
    # markers_arr = markers_arr[:len(expirements)]
    for ind, marker in enumerate(marker_arr):
        if ind in [0, 6]:
            markersize = 11
        else:
            markersize = 8
        plt.plot(list(single_model_res_arr.values())[ind], list(ensemble_model_res_arr.values())[ind], marker[0],
                 label=f"{list(single_model_res_arr.keys())[ind]}", markersize=markersize, mfc=marker[1], mec=marker[1])
    # plt.scatter(single_model_res_arr.values(), ensemble_model_res_arr.values())
    plt.plot([0.0, 0.6], [0.0, 0.6], ':', lw=2)
    plt.legend()
    plt.ylabel('Ensemble model')
    plt.xlabel('Single model')
    plt.title('Ensemble Vs Single')
    plt.xlim(0.06, 0.6)
    plt.ylim(0.06, 0.6)
    plt.show()
    plt.savefig('results/transfer_learning/compare_ensemble.png')
    a=0


def tl_veriations_hit_map():
    import numpy as np
    from pandas import DataFrame

    idx = ['aaa', 'bbb', 'ccc', 'ddd', 'eee']
    cols = list('ABCD')
    df = DataFrame(abs(np.random.randn(5, 4)), index=idx, columns=cols)

    # _r reverses the normal order of the color map 'RdYlGn'
    # sns.heatmap(df, cmap='RdYlGn_r', linewidths=0.5, annot=True)
    # plt.show()


    expirements_df = {}
    for exp in expirements:
        path = f'results/transfer_learning/{exp}/avg_results.csv'
        avg_res_df = pd.read_csv(path, index_col=0)
        expirements_df[exp] = avg_res_df


    enzyme_res_dfs = {}
    for enzyme in enzymes:
        columns = rows_ensemble if enzyme == 'wt' else rows_ensemble[:-1]
        zero_data = np.zeros(shape=(len(expirements), len(columns)))
        final_res = pd.DataFrame(zero_data, index=expirements, columns=columns)
        for exp in expirements:
            df = expirements_df[exp]
            enzyme_res = abs(df[enzyme])
            final_res.loc[exp] = enzyme_res

        enzyme_res_dfs[enzyme] = final_res

        fig = plt.figure()
        heatmap = sns.heatmap(final_res, cmap='RdYlGn_r', linewidths=0.5, annot=True, fmt='.3g')
        plt.title(f'Pre train dataset - {enzyme}')
        plt.show()
        # heatmap.get_figure().savefig(f'results/transfer_learning/{enzyme}_final_res.png')

        # plt.show()
        # final_res.to_csv(f'results/transfer_learning/{enzyme}_final_res.csv')

    a=0


def tl_veriations_bar_plot():
    import numpy as np

    expirements_df = {}
    err_df = {}
    for exp in expirements:
        path = f'results/transfer_learning/{exp}/avg_results.csv'
        avg_res_df = pd.read_csv(path, index_col=0)
        expirements_df[exp] = avg_res_df

        path = f'results/transfer_learning/{exp}/std_results.csv'
        std_res_df = pd.read_csv(path, index_col=0)
        err_df[exp] = std_res_df


    enzyme_res_dfs = {}
    for enzyme in enzymes:
        columns = rows_ensemble if enzyme in ['wt', 'multi_task'] else rows_ensemble[:-1]
        zero_data = np.zeros(shape=(len(expirements), len(columns)))
        final_res = pd.DataFrame(zero_data, index=expirements, columns=columns)
        final_err = final_res.copy(deep=True)
        for exp in expirements:
            df = expirements_df[exp]
            enzyme_res = abs(df[enzyme])
            final_res.loc[exp] = enzyme_res

            df = err_df[exp]
            enzyme_res = abs(df[enzyme])
            final_err.loc[exp] = enzyme_res


        enzyme_res_dfs[enzyme] = final_res

        columns = rows if enzyme in ['wt', 'multi_task'] else rows[:-1]
        final_res.set_axis(columns, axis=1, inplace=True)
        final_err.set_axis(columns, axis=1, inplace=True)

        fig = plt.figure()
        ax = final_res.plot.bar(yerr=final_err, capsize=3, rot=0, width=0.8)
        ax.set_ylabel('Spearman')
        ax.set_xlabel('Dataset')
        ax.set_xticklabels(expirements, rotation=45, ha="right")
        leg = ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.9), shadow=True, ncol=2)
        leg.set_alpha(0.1)
        # plt.ylim(0.0, 0.75)
        # plt.title(f'Pre train dataset - {enzyme}')
        plt.show()
        # plt.savefig(f'results/transfer_learning/{enzyme}_final_bar_res.png')
        a=0
        if T7:
            final_res.to_csv(f'results/transfer_learning/{enzyme}_T7_final_res.csv')
            final_err.to_csv(f'results/transfer_learning/{enzyme}_T7_final_err.csv')
        else:
            final_res.to_csv(f'results/transfer_learning/{enzyme}_final_res.csv')
            final_err.to_csv(f'results/transfer_learning/{enzyme}_final_err.csv')

    a=0

def tl_veriations_vs_crispron_bar_plot():
    if T7:
        final_res = pd.read_csv(f'results/transfer_learning/multi_task_T7_final_res.csv', index_col=0)
        final_err = pd.read_csv(f'results/transfer_learning/multi_task_T7_final_err.csv', index_col=0)
        crispron_final_res = pd.read_csv(f'results/transfer_learning/crispron_T7_final_res.csv', index_col=0)
        crispron_final_err = pd.read_csv(f'results/transfer_learning/crispron_T7_final_err.csv', index_col=0)
    else:
        final_res = pd.read_csv(f'results/transfer_learning/multi_task_final_res.csv', index_col=0)
        final_err = pd.read_csv(f'results/transfer_learning/multi_task_final_err.csv', index_col=0)
        crispron_final_res = pd.read_csv(f'results/transfer_learning/crispron_final_res.csv', index_col=0)
        crispron_final_err = pd.read_csv(f'results/transfer_learning/crispron_final_err.csv', index_col=0)

    fig, ax = plt.subplots()
    fig.set_size_inches(15, 5)
    reactes = []
    ind = np.arange(final_res.shape[0]) * 9 -5
    distance = 0.7
    i = 0
    train_types = rows
    colors = {'gl_tl': 'tab:blue', 'full_tl': 'tab:orange', 'no_em_tl': 'tab:green', 'LL_tl': 'tab:red', 'no_tl': 'tab:purple', 'no_pre_train': 'tab:brown'}
    for (train_type, values), (err_train_type, err_values) in zip(final_res.items(), final_err.items()):
        values = values.to_numpy()
        err_values = err_values.to_numpy()
        react = ax.bar(ind + i*distance, values, distance, label='Single-Task', color=colors[train_type], yerr=err_values, capsize=1.5)
        i+=1
        reactes.append(react)

    j=0
    for (train_type, values), (err_train_type, err_values) in zip(crispron_final_res.items(), crispron_final_err.items()):
        values = values.to_numpy()
        err_values = err_values.to_numpy()
        react = ax.bar(ind + i*distance, values, distance, label='Single-Task', color=colors[train_type], yerr=err_values, capsize=1.5, hatch = 'O')
        i+=1
        j+=1
        reactes.append(react)

    leg2 = ax.legend(reactes[0:6], train_types, bbox_to_anchor=(0.12, 0.45), fontsize=12) # For T7
    leg3 = ax.legend([reactes[1], reactes[7]], ['DeepHF', 'CRISPROn'],  bbox_to_anchor=(0.13, 0.83))
    # leg2 = ax.legend(reactes[0:6], train_types, bbox_to_anchor=(0.88, 0.4), fontsize=12)
    # leg3 = ax.legend([reactes[1], reactes[7]], ['DeepHF', 'CRISPROn'],  bbox_to_anchor=(0.77, 0.8), fontsize=12)
    ax.add_artist(leg2)

    plt.ylabel('Spearman', fontsize=12)
    plt.xlabel('Dataset', fontsize=12)
    plt.xticks(ind + 0.15, expirements, rotation=45, ha="right", fontsize=12)
    plt.show()
    a=0




def final_table():
    enzyme = 'multi_task'
    models = list(pd.read_csv(f'data/tl_train/U6T7/{expirements[0]}/results.csv', index_col=0).keys())
    final_res = pd.DataFrame(index=expirements, columns=['DeepHF', 'DeepCRISTL', 'CRISPRon'] + models)
    final_err = final_res.copy(deep=True)

    for exp in expirements:
        if exp == 'leenay':
            results = {}
            results_err = {}
        else:
            exp_df = pd.read_csv(f'data/tl_train/U6T7/{exp}/results.csv', index_col=0)
            results = abs(exp_df.mean(axis=0))
            results_err = exp_df.std(axis=0)

        deep_cristl_df = pd.read_csv(f'results/transfer_learning/{exp}/avg_results.csv', index_col=0)
        deephf = deep_cristl_df['wt']['LL_tl']
        deep_cristl = deep_cristl_df[enzyme]['gl_tl_ensemble']

        deep_cristl_err_df = pd.read_csv(f'results/transfer_learning/{exp}/std_results.csv', index_col=0)
        deephf_err = deep_cristl_err_df['wt']['LL_tl']
        deep_cristl_err = deep_cristl_err_df[enzyme]['gl_tl_ensemble']

        crispron_df = pd.read_csv(f'../CRISPRon/results/transfer_learning/{exp}/avg_results.csv', index_col=0)
        # crispron_sets_res = crispron_df['gl_tl_ensemble']
        # res = 0
        # for val in crispron_sets_res:
        #     val = val.split(' ')[0]
        #     res+=float(val)
        # crispron = res/5
        crispron = crispron_df['avg_result']['full_tl_ensemble']
        crispron_err = crispron_df['std_result']['full_tl_ensemble']

        results['DeepHF'] = deephf
        results['DeepCRISTL'] = deep_cristl
        results['CRISPRon'] = crispron

        if exp != 'leenay':
            results = results.apply(lambda x: float('%.4f' % x))
            results = results.to_dict()
        final_res.loc[exp] = results

        results_err['DeepHF'] = deephf_err
        results_err['DeepCRISTL'] = deep_cristl_err
        results_err['CRISPRon'] = crispron_err

        if exp != 'leenay':
            results_err = results_err.apply(lambda x: float('%.4f' % x))
            results_err = results_err.to_dict()
        final_err.loc[exp] = results_err

    final_res.to_csv('results/transfer_learning/all_experiments_table.csv')

    final_res = final_res[['DeepCRISTL', 'DeepHF', 'CRISPRon', 'wuCrispr', 'wangOrig', 'doench', 'chariRank']]
    final_res.rename(columns={'wangOrig': 'wang score', 'doench': 'doench score', 'chariRank':'chari score'},inplace=True)
    final_err = final_err[['DeepCRISTL', 'DeepHF', 'CRISPRon', 'wuCrispr', 'wangOrig', 'doench', 'chariRank']]
    final_err.rename(columns={'wangOrig': 'wang score', 'doench': 'doench score', 'chariRank':'chari score'},inplace=True)

    fig = plt.figure()

    ax = final_res.plot.bar(yerr=final_err, capsize=1.5, rot=0, width=0.8, figsize=(15, 5))
    ax.set_xticklabels(expirements, rotation=45, ha="right", fontsize=12)
    ax.set_ylabel('Spearman', fontsize=12)
    ax.set_xlabel('Dataset', fontsize=12)

    leg = ax.legend(loc="upper left", bbox_to_anchor=(0.12, 0.999), shadow=False, fontsize=12, ncol=3)
    # plt.title('models comparison')

    plt.show()

    a=0


# def interpertate(config):
#     config.train_type = 'full_tl'
#     model = load_model('models/transfer_learning/xu2015TrainHl60/set0/DeepHF_old/esp/full_tl/model_0/model')
#     from scripts import models_util
#     from scripts_tl import data_handler_tl as dh_tl
#     from scripts_tl import training_util_tl
#     DataHandler = dh_tl.get_data(config, 0)
#     config.save_model = False
#     model, callback_list = models_util.load_pre_train_model(config, DataHandler)
#     # model.fit([DataHandler['X_train'], DataHandler['X_biofeat_train']],
#     #           DataHandler['y_train'],
#     #           batch_size=config.batch_size,
#     #           epochs=1,
#     #           verbose=1,
#     #           validation_data=([DataHandler['X_valid'], DataHandler['X_biofeat_valid']], DataHandler['y_valid']),
#     #           shuffle=True,
#     #           callbacks=callback_list,
#     #           )
#     explainer = shap.DeepExplainer(model, [DataHandler['X_train'], DataHandler['X_biofeat_train']])
#     shap_values = explainer.shap_values([DataHandler['X_test'], DataHandler['X_biofeat_test']])
#     shap.force_plot(explainer.expected_value[0], shap_values[0][0], x_test_words[0])
#     a=0


def get_learning_curve(config):

    # This method is for plotting the learning curve of gradual learning transfer learning
    DataHandler = dh_tl.get_data(config, 0)
    config.save_model = False
    config.train_type = 'LL_tl'
    model, callback_list = models_util.load_pre_train_model(config, DataHandler)

    config.epochs = 100
    LL_history = training_util_tl.train_model(config, DataHandler, model, callback_list)

    config.set = 0
    config.train_type = 'gl_tl'
    model, callback_list = models_util.load_pre_train_model(config, DataHandler)
    gl_history = training_util_tl.train_model(config, DataHandler, model, callback_list)

    val_loss_curve = LL_history.history['val_loss'] + gl_history.history['val_loss']
    loss_curve = LL_history.history['loss'] + gl_history.history['loss']

    fig = plt.figure()
    plt.plot(range(1, len(loss_curve)+1), val_loss_curve, label='val_loss')
    plt.plot(range(1, len(loss_curve)+1), loss_curve, label='loss')


    # Add ranges of LL_tl, gl_tl and early stoppings
    LL_tl_range = len(LL_history.history['val_loss'])
    LL_tl_early_stopping = LL_history.history['val_loss'].index(min(LL_history.history['val_loss'])) + 1
    gl_tl_range = len(gl_history.history['val_loss']) + LL_tl_range
    gl_tl_early_stopping = gl_history.history['val_loss'].index(min(gl_history.history['val_loss'])) + LL_tl_range + 1

    plt.axvspan(0, LL_tl_early_stopping, color='red', alpha=1)
    plt.axvspan(LL_tl_early_stopping, LL_tl_range, color='red', alpha=0.5)
    plt.axvspan(LL_tl_range, gl_tl_early_stopping, color='green', alpha=1)
    plt.axvspan(gl_tl_early_stopping, gl_tl_range, color='green', alpha=0.5)
    plt.legend()
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('loss function curve using gradual learning')
    plt.show()


def test_KBM7_model(config):
    config.save_model = False
    config.tl_data_category = 'U6T7'

    train_types = ['LL_tl', 'gl_tl']

    for train_type in train_types:
        # Loading final ensemble model
        print(f'Loading final model of {train_type}')
        start = time.time()
        config.tl_data = 'xu2015TrainKbm7'
        config.train_type = train_type

        final_model = []
        for set in range(3, 4): #TODO - change to 5
            print(f'Loading set {set} models')
            config.set = set
            all_set_models = ensemble_util_tl.load_all_models(config)
            final_model = final_model + all_set_models

        end = time.time()
        print(f'Finished loading ensemble model of {train_type} in {end - start} sec')


        for exp in expirements:
            for set in range(1): #TODO - change to 5
                config.set = set
                config.tl_data = exp
                DataHandler = dh_tl.get_data(config, set, verbose=0)

                spearmanr = ensemble_util_tl.test_means(config, final_model, DataHandler, verbose=0)
                print(f'Expirement: {exp}, Spearman: {spearmanr}')



def ensemble_models_curve(config):
    models_dir = f'models/transfer_learning/doench2014-Hs/set0/DeepHF_old/multi_task/gl_tl/'

    all_models = []
    model_ind = 0
    model_path = models_dir + 'model_0/model'
    while os.path.exists(model_path):
        print(f'Loading model_{model_ind}')
        model = load_model(model_path)
        model_ind += 1
        model_path = models_dir + f'model_{model_ind}/model'
    a=0