import numpy as np
import os
import pandas as pd


def predict(all_models, DataHandler):
    predictions = []
    test_input = [DataHandler['X_test'], DataHandler['dg_test']]
    for ind, model in enumerate(all_models):
        print(f'Genating predictions for model_{ind}')
        test_prediction = model.predict(test_input)
        predictions.append(test_prediction)

    final_pred = np.zeros((test_prediction.shape[0], 1))
    for pred in predictions:
        final_pred += pred
    final_pred /= len(predictions)

    return final_pred

def save_prediction_file(config, final_pred):
    input_df = pd.read_csv(f'tool data/input/{config.input_file}.csv')

    print('Saving prediction file')
    dir_path = f'tool data/output/{config.input_file}/'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    df = pd.DataFrame()

    df['30mer'] = input_df['30mer']
    df['predicted'] = final_pred

    df.to_csv(dir_path + 'predicted_by_' + config.model_to_use + '.csv', index=False)



