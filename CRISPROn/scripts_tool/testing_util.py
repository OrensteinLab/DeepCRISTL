import scipy as sp
import numpy as np



def get_spearmanr(all_models, DataHandler):

    predictions = []
    test_input = [DataHandler['X_test'], DataHandler['dg_test']]
    test_true_label = DataHandler['y_test']
    for ind, model in enumerate(all_models):
        test_prediction = model.predict(test_input)
        predictions.append(test_prediction)

    finall_pred = np.zeros((test_prediction.shape[0], 1))
    for pred in predictions:
        finall_pred += pred
    finall_pred /= len(predictions)
    spearmanr = sp.stats.spearmanr(test_true_label, finall_pred)
    # get speramanr and p value

    return spearmanr[0], spearmanr[1]