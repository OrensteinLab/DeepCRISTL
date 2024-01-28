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
import numpy as np
# from scripts_tl import data_handler_tl as dh_tl
# from scripts_tl import training_util_tl
# from scripts import models_util
# from scripts_tl import ensemble_util_tl


expirements = ['xu2015TrainHl60', 'chari2015Train293T', 'hart2016-Rpe1Avg', 'hart2016-Hct1162lib1Avg',
               'hart2016-HelaLib1Avg', 'hart2016-HelaLib2Avg','xu2015TrainKbm7', 'doench2014-Hs' , 'doench2014-Mm',
               'doench2016_hg19', 'leenay']


T7_expirements = ['eschstruth', 'varshney2015', 'gagnon2014', 'shkumatavaPerrine', 'shkumatavaAngelo', 'shkumatavaOthers', 'teboulVivo_mm9', 'morenoMateos2015']
T7 = False

if T7:
    expirements = T7_expirements

crispr_il_expirements = ['HOP.Cas9.U937_Human_Monocytes_clean_features_eff']

rows_ensemble = ['gl_tl_ensemble','full_tl_ensemble','LL_tl_ensemble', 'no_tl_ensemble', 'no_pre_train_ensemble']
rows = ['gl_tl','full_tl','LL_tl', 'no_tl', 'no_pre_train']


def postprocess(config):
    calc_avg_res(config)
    # compare_ensemble()
    # tl_veriations_hit_map()
    # tl_veriations_bar_plot()

def calc_avg_res(config):
    exps = expirements if config.tl_data_category == 'U6T7' else crispr_il_expirements

    for exp in exps:
        print(exp)
        sets_df_array = []
        avg_res_df = pd.DataFrame(index=rows_ensemble, columns=['avg_result', 'std_result'])
        path = f'results/transfer_learning/{exp}/'
        set_res_df = pd.read_csv(path + 'results.csv', index_col=0)
        for type in rows_ensemble:
            scores = set_res_df.loc[:, type].to_numpy()
            avg_score = 0
            score_arr = []
            for score in scores:
                if (pd.isna(score) or 'nan' in score):
                    val = 0
                else:
                    val = float(score.split(' ')[0])
                score_arr.append(val)
            if val<0:
                a=0
            avg_score = np.average(score_arr)
            std_score = np.std(score_arr)
            avg_res_df['avg_result'][type] = avg_score
            avg_res_df['std_result'][type] = std_score

            a=0
        # for set in range(5):
        #     set_path = path + f'set{set}/results.csv'
        #     set_res_df = pd.read_csv(set_path, index_col=0)
        #     sets_df_array.append(set_res_df)


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
    for exp in expirements:
        path = f'results/transfer_learning/{exp}/avg_results.csv'
        avg_res_df = pd.read_csv(path, index_col=0)
        expirements_df[exp] = avg_res_df


    enzyme_res_dfs = {}

    columns = rows_ensemble
    zero_data = np.zeros(shape=(len(expirements), len(columns)))
    final_res = pd.DataFrame(zero_data, index=expirements, columns=columns)
    final_error = final_res.copy(deep=True)
    for exp in expirements:
        df = expirements_df[exp]
        enzyme_res = abs(df['avg_result'])
        final_res.loc[exp] = enzyme_res
        error = df['std_result']
        final_error.loc[exp] = error


    enzyme_res_dfs['avg_result'] = final_res

    columns = rows
    final_res.set_axis(columns, axis=1, inplace=True)
    final_error.set_axis(columns, axis=1, inplace=True)

    fig = plt.figure()
    ax = final_res.plot.bar(yerr=final_error, capsize=3, rot=0, width=0.7)
    ax.set_ylabel('Spearman')
    ax.set_xlabel('Dataset')
    ax.set_xticklabels(expirements, rotation=45, ha="right")
    leg = ax.legend(loc="upper left", bbox_to_anchor=(0.3, 0.2), shadow=True, ncol=3)
    leg.set_alpha(0.1)
    plt.ylim(0.0, 0.75)
    # plt.title(f'Pre train dataset - {enzyme}')
    plt.show()
    # plt.savefig(f'results/transfer_learning/{enzyme}_final_bar_res.png')
    a=0
    if T7:
        final_res.to_csv(f'results/transfer_learning/crispron_T7_final_res.csv')
        final_error.to_csv(f'results/transfer_learning/crispron_T7_final_err.csv')
    else:
        final_res.to_csv(f'results/transfer_learning/crispron_final_res.csv')
        final_error.to_csv(f'results/transfer_learning/crispron_final_err.csv')

    a=0


def final_table():
    enzyme = 'multi_task'
    models = list(pd.read_csv(f'data/tl_train/U6T7/{expirements[0]}/results.csv', index_col=0).keys())
    final_res = pd.DataFrame(index=expirements, columns=['DeepHF', 'DeepCRISTL', 'CRISPRon'] + models)

    for exp in expirements:
        exp_df = pd.read_csv(f'data/tl_train/U6T7/{exp}/results.csv', index_col=0)
        results = abs(exp_df.mean(axis=0))

        deep_cristl_df = pd.read_csv(f'results/transfer_learning/{exp}/avg_results.csv', index_col=0)
        deephf = deep_cristl_df['wt']['LL_tl']
        deep_cristl = deep_cristl_df[enzyme]['gl_tl_ensemble']

        crispron_df = pd.read_csv(f'../CRISPRon/results/transfer_learning/{exp}/results.csv', index_col=0)
        crispron_sets_res = crispron_df['gl_tl_ensemble']
        res = 0
        for val in crispron_sets_res:
            val = val.split(' ')[0]
            res+=float(val)
        crispron = res/5

        results['DeepHF'] = deephf
        results['DeepCRISTL'] = deep_cristl
        results['CRISPRon'] = crispron

        results = results.apply(lambda x: float('%.4f' % x))
        results = results.to_dict()
        final_res.loc[exp] = results

    final_res.to_csv('results/transfer_learning/all_experiments_table.csv')

    final_res = final_res[['DeepCRISTL', 'DeepHF', 'CRISPRon', 'wuCrispr', 'wangOrig', 'doench', 'chariRank']]
    final_res.rename(columns={'wangOrig': 'wang score', 'doench': 'doench score', 'chariRank':'chari score'},inplace=True)
    fig = plt.figure()
    ax = final_res.plot.bar(rot=0)
    ax.set_xticklabels(expirements, rotation=60, ha="right", fontsize=10)
    leg = ax.legend(loc="upper left", bbox_to_anchor=(0.15, 1.1), shadow=True, fontsize=6)
    plt.title('models comparison')
    plt.show()

    a=0


def interpertate(config):
    config.train_type = 'full_tl'
    model = load_model('models/transfer_learning/xu2015TrainHl60/set0/DeepHF_old/esp/full_tl/model_0/model')
    from scripts import models_util
    from scripts_tl import data_handler_tl as dh_tl
    from scripts_tl import training_util_tl
    DataHandler = dh_tl.get_data(config, 0)
    config.save_model = False
    model, callback_list = models_util.load_pre_train_model(config, DataHandler)
    # model.fit([DataHandler['X_train'], DataHandler['X_biofeat_train']],
    #           DataHandler['y_train'],
    #           batch_size=config.batch_size,
    #           epochs=1,
    #           verbose=1,
    #           validation_data=([DataHandler['X_valid'], DataHandler['X_biofeat_valid']], DataHandler['y_valid']),
    #           shuffle=True,
    #           callbacks=callback_list,
    #           )
    explainer = shap.DeepExplainer(model, [DataHandler['X_train'], DataHandler['X_biofeat_train']])
    shap_values = explainer.shap_values([DataHandler['X_test'], DataHandler['X_biofeat_test']])
    shap.force_plot(explainer.expected_value[0], shap_values[0][0], x_test_words[0])
    a=0


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


