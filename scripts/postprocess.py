from scripts import data_handler as dh
from scripts import ensemble_util
import numpy as np
import scipy as sp
import os
from keras.models import load_model, Model
import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import pickle

def postprocess(config):
    # final_results()
    ensemble_vs_num_of_models(config)


def final_results():
    ind = np.arange(3)
    avg_bar1 = (0.873, 0.871, 0.8603)
    avg_bar2 = (0.8784, 0.8743, 0.8652)
    avg_bar3 = (0.8805, 0.87832, 0.8674)
    avg_bar4 = (0.8872, 0.8842, 0.8758)

    fig, ax = plt.subplots()

    rects1 = ax.bar(ind, avg_bar1, 0.15, label='Single-Task', color='r')
    rects2 = ax.bar(ind + 0.15, avg_bar2, 0.15, label='Multi-Task', color='g')
    rects3 = ax.bar(ind + 0.30, avg_bar3, 0.15, label='Single-Model', color='r', hatch = 'O')
    rects4 = ax.bar(ind + 0.45, avg_bar4, 0.15, label='Ensemble-Model', color='g', hatch = 'O')

    plt.xlabel('Enzyme')
    plt.ylabel('Spearman')
    plt.xticks(ind + 0.15, ('WT', 'ESP', 'HF'))

    leg2 = ax.legend([rects1, rects2], ['Single-Task', 'Multi-Task'], bbox_to_anchor=(0.65, 0.92))
    leg3 = ax.legend([rects1, rects3], ['Single-Model', 'Ensemble-Model'],  bbox_to_anchor=(0.65, 0.77))
    ax.add_artist(leg2)

    plt.ylim(0.855, 0.89)

    plt.tight_layout()
    plt.show()
    exit()



def ensemble_vs_num_of_models(config):
    enzymes = ['wt', 'esp', 'hf', 'multi_task']
    # enzymes = ['multi_task']

    if os.path.exists('results/pre_train/spearmans.pkl'):
        with open('results/pre_train/spearmans.pkl', "rb") as fp:
            spearmans = pickle.load(fp)
    else:
        spearmans = {}
        for enzyme in enzymes:
            config.enzyme = enzyme

            DataHandler = dh.get_data(config)
            config.save_model = False
            spearman_arr = test_means(config, DataHandler)
            spearmans[enzyme] = spearman_arr

            with open('results/pre_train/spearmans.pkl', "wb") as fp:
                pickle.dump(spearmans, fp)
    plot_spearman_curve(spearmans)

def test_means(config, DataHandler):
    print(f'\nTesting {config.enzyme} models:')
    models_dir = 'models/transfer_learning/' if config.transfer_learning else 'models/pre_train/'
    models_dir += f'{config.pre_train_data}/{config.enzyme}/'

    if config.enzyme != 'multi_task':
        test_input = [DataHandler['test'].enzymes_seq[config.enzyme].X, DataHandler['test'].enzymes_seq[config.enzyme].X_biofeat]
        test_true_label = DataHandler['test'].enzymes_seq[config.enzyme].y

    all_models = []
    model_ind = 0
    model_path = models_dir + 'model_0/model'

    if config.enzyme == 'multi_task':
        predictions = {'wt': [], 'esp': [], 'hf': []}
    else:
        predictions = []

    # Receiving predictions
    while os.path.exists(model_path):
        # if model_ind == 2:
        #     break
        print(f'Loading model_{model_ind}')
        model = load_model(model_path)


        if config.enzyme == 'multi_task':
            enzymes = ['wt', 'esp', 'hf']
            spearman_result = {}
            for enzyme in enzymes:
                test_input = [DataHandler['test'].enzymes_seq[enzyme].X, DataHandler['test'].enzymes_seq[enzyme].X_biofeat]
                test_prediction = model.predict(test_input)
                predictions[enzyme].append(test_prediction)
        else:
            test_prediction = model.predict(test_input)
            predictions.append(test_prediction)

        keras.backend.clear_session()
        model_ind += 1
        model_path = models_dir + f'model_{model_ind}/model'



    # Calculating spearman
    if config.enzyme == 'multi_task':
        spearmans = {'wt': [], 'esp': [], 'hf': []}
        for enzyme in enzymes:

            enz_predictions = np.array(predictions[enzyme])
            enz_predictions = np.squeeze(enz_predictions)
            final_preds = np.zeros((test_prediction.shape[0], len(enz_predictions)))
            for i in range(len(enz_predictions)):
                preds = enz_predictions[:i + 1]
                if i == 0:
                    final_preds[:, i] = preds
                else:
                    final_preds[:, i] = np.mean(preds, axis=0)

            test_true_label = DataHandler['test'].enzymes_seq[enzyme].y
            for i in range(final_preds.shape[1]):
                pred = final_preds[:, i]
                spearman = sp.stats.spearmanr(test_true_label, pred)[0]
                print(spearman)
                spearmans[enzyme].append(spearman)
    else:
        predictions = np.array(predictions)
        predictions = np.squeeze(predictions)
        final_preds =  np.zeros((test_prediction.shape[0], len(predictions)))
        for i in range(len(predictions)):
            preds = predictions[:i+1]
            if i == 0:
                final_preds[:, i] = preds
            else:
                final_preds[:, i] = np.mean(preds, axis=0)

        spearmans = []
        for i in range(final_preds.shape[1]):
            pred = final_preds[:, i]
            spearman = sp.stats.spearmanr(test_true_label, pred)[0]
            print(spearman)
            spearmans.append(spearman)

    return spearmans
    # if config.enzyme == 'multi_task':
    #     enzymes = ['wt', 'esp', 'hf']
    #     spearman_result = {}
    #     # for enzyme in enzymes:
    #     #     test_input = [DataHandler['test'].enzymes_seq[enzyme].X, DataHandler['test'].enzymes_seq[enzyme].X_biofeat]
    #     #     test_true_label = DataHandler['test'].enzymes_seq[enzyme].y
    #     #     predictions = []
    #     #     for ind, model in enumerate(all_models):
    #     #         print(f'Testing model_{ind} with {enzyme}')
    #     #         test_prediction = model.predict(test_input)
    #     #         predictions.append(test_prediction)
    #     #
    #     #     finall_pred = np.zeros((test_prediction.shape[0], 1))
    #     #     for pred in predictions:
    #     #         finall_pred += pred
    #     #     finall_pred /= len(predictions)
    #     #     spearmanr = sp.stats.spearmanr(test_true_label, finall_pred)
    #     #     spearman_result[enzyme] = spearmanr
    #     # for enzyme in enzymes:
    #     #     print(f'Enzyme: {enzyme}, Spearman: {spearman_result[enzyme]}')

def plot_spearman_curve(spearmans):
    enzymes = ['wt', 'esp', 'hf']

    labels = {'wt': 'WT', 'esp': 'ESP', 'hf': 'HF'}
    colors = {'wt': 'r', 'esp': 'g', 'hf': 'b'}
    enzyme_legend = []
    sing_vs_multi_legend = []

    fig, ax = plt.subplots()

    for enzyme, spearman_arr in spearmans.items():
        if enzyme == 'multi_task':
            for enzyme, spearman_enz in spearman_arr.items():
                x = np.arange(1, len(spearman_enz) + 1)
                label = labels[enzyme]
                react, = plt.plot(x, spearman_enz, 'x-', label=f'Multi-Task - {label}', color=colors[enzyme])
                sing_vs_multi_legend.append(react)
        else:
            label = labels[enzyme]
            x = np.arange(1, len(spearman_arr)+1)
            react, = plt.plot(x, spearman_arr, 'o-',  label=label, color=colors[enzyme])
            enzyme_legend.append(react)


    leg2 = plt.legend(enzyme_legend, enzymes, bbox_to_anchor=(0.65, 0.22))
    leg3 = plt.legend([enzyme_legend[0], sing_vs_multi_legend[0]], ['Single-task', 'Multi-task'],  bbox_to_anchor=(0.9, 0.22))
    plt.gca().add_artist(leg2)

    # plt.legend(bbox_to_anchor=(0.3, 0.4))
    plt.ylabel('Spearman')
    plt.xlabel('Number of models')
    plt.xticks(np.arange(1, 21))
    plt.title('Ensemble model spearman result Vs number of models')
    plt.show()




