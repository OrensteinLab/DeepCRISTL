

from scripts_tool import data_handler as dh
from scripts_tool import utils
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def generate_heatmap(config, from_pickle=False):
    import pickle
    from scripts_tool import models_util
    from scripts_tool import testing_util



    models = utils.get_all_models()
    models.sort()
    datasets = utils.get_all_datasets()
    datasets.sort()
    models.append('no_transfer_learning')
    models_spearmans = {}

    if not from_pickle:
        print('Calculating spearmans')
        for model in models:
            print(f'\n\n\nCalculating spearmans for model trained on {model}')
            print('-----------------------------------\n')
            if model == 'no_transfer_learning':
                ensemble_models = models_util.load_no_tl_models()
            else:
                config.model_to_use = model
                ensemble_models = models_util.load_tl_models(config)
            models_spearmans[model] = {}
            for dataset in datasets:
                print(f'Calculating spearmans on {dataset}')
                DataHandler = dh.get_data_from_dataset(dataset)
                spearmanr, pvalue = testing_util.get_spearmanr(ensemble_models, DataHandler) #TODO: add p value to the output?
                models_spearmans[model][dataset] = spearmanr

        print('Spearmans:')
        for model in models_spearmans:
            print(f'\n\n\n{model}')
            print('-----------------------------------\n\n')
            for dataset in models_spearmans[model]:
                print(f'{dataset}: {models_spearmans[model][dataset]}')


        numpy_array = []
        for model in models_spearmans:
            numpy_array.append([models_spearmans[model][dataset] for dataset in models_spearmans[model]])
            

        # save the the numpy array to a pickle file
        with open('tool data/output/spearmans.pkl', 'wb') as f:
            pickle.dump(numpy_array, f)   
    else:
        print('Loading spearmans from pickle')  

    #load the numpy array from the pickle file
    with open('tool data/output/spearmans.pkl', 'rb') as f:
        numpy_array = pickle.load(f)   


    # make into a numpy array
    numpy_array = np.array(numpy_array)

    # make into a dataframe
    dataframe = pd.DataFrame(numpy_array, columns=datasets, index=models)

    generate_one_heatmap(dataframe, numpy_array, models, datasets, 'rc', order_rows=True, order_columns=True)
    generate_one_heatmap(dataframe, numpy_array, models, datasets, 'r', order_rows=True, order_columns=False)
    generate_one_heatmap(dataframe, numpy_array, models, datasets, 'c', order_rows=False, order_columns=True)
    generate_one_heatmap(dataframe, numpy_array, models, datasets, 'none', order_rows=False, order_columns=False)





    
def get_reordered(numpy_array, row_order, column_order, models, datasets):
    # reorder the rows in the numpy array and the models
    new_numpy_array = numpy_array[row_order]
    new_models = [models[i] for i in row_order]

    # reorder the columns in the numpy array and the datasets
    new_numpy_array = new_numpy_array[:, column_order]
    new_datasets = [datasets[i] for i in column_order]

    return new_numpy_array, new_models, new_datasets


def generate_clustermap(dataframe, order_rows=True, order_columns=True, name='clustermap'):
    
    # create a clustermap
    cluster_grid = sns.clustermap(dataframe, cmap='viridis', method='average', col_cluster=order_columns, row_cluster=order_rows)     
    # save the the clustermap to a file

    PATH_FOR_CLUSTERMAP = 'tool data/output/' + name + '_clustermap.png'
    cluster_grid.savefig(PATH_FOR_CLUSTERMAP)

    print('Clustermap saved to ' + PATH_FOR_CLUSTERMAP)
    if order_rows:
        row_order = cluster_grid.dendrogram_row.reordered_ind
    else:
        row_order = list(range(len(dataframe.index)))
    
    if order_columns:
        column_roder = cluster_grid.dendrogram_col.reordered_ind
    else:
        column_roder = list(range(len(dataframe.columns)))

    return row_order, column_roder





def generate_one_heatmap(dataframe, numpy_array, models, datasets, name, order_rows=True, order_columns=True):

    row_order, column_roder = generate_clustermap(dataframe, order_rows=order_rows, order_columns=order_columns,name=name)
    numpy_array, models, datasets = get_reordered(numpy_array, row_order, column_roder, models, datasets)

    # get the index of the model 'no_transfer_learning'
    no_tl_index = models.index('no_transfer_learning')

    # make sure it is the last row
    numpy_array = np.concatenate([numpy_array[:no_tl_index], numpy_array[no_tl_index+1:], numpy_array[no_tl_index:no_tl_index+1]], axis=0)
    models = models[:no_tl_index] + models[no_tl_index+1:] + models[no_tl_index:no_tl_index+1]



    print('Plotting heatmap for ' + name )

    # make a bigger plot
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)


    
    # Create a heatmap using Matplotlib's imshow function
    heatmap = ax.imshow(numpy_array, cmap='coolwarm', interpolation='nearest')


    # Add the values to the heatmap
    for i in range(len(models)):
        for j in range(len(datasets)):
            text = ax.text(j, i, format(numpy_array[i][j], '.2f'),
                        ha="center", va="center", color="black", fontsize=14)

    # Add colorbar to the right of the heatmap
    plt.colorbar(heatmap)

    # Set labels for the axes
    plt.xlabel('Dataset', fontsize=16)
    plt.ylabel('Model', fontsize=16)

    # Set the title for the heatmap
    plt.title('Spearmans R', fontsize=20)

    # Set labels for each column base on the dataset
    plt.xticks([i for i in range(len(datasets))], datasets, rotation=90)

    # Set labels for each row base on the model
    plt.yticks([i for i in range(len(models))], models)

    # Save the heatmap to a file TODO: change it to output again
    #PATH_FOR_HEATMAP = 'tool data/output/heatmap.png'
    PATH_FOR_HEATMAP = 'tool data/output/' + name + '_heatmap.png'
    plt.savefig(PATH_FOR_HEATMAP)

    print('Heatmap saved to ' + PATH_FOR_HEATMAP)

