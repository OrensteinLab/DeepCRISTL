

from scripts_tool import data_handler as dh
from scripts_tool import utils
import numpy as np
import pandas as pd
import seaborn as sns

def generate_heatmap(config, from_pickle=False):
    import pickle
    from scripts_tool import models_util
    from scripts_tool import testing_util
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt


    models_and_datasets = utils.get_all_models()
    models_and_datasets.sort()
    datasets = models_and_datasets
    models = models_and_datasets + ['no_transfer_learning']
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

    # create a clustermap
    cluster_grid = sns.clustermap(dataframe, cmap='viridis', method='average', col_cluster=True, row_cluster=True)        

    # save the the clustermap to a file

    PATH_FOR_CLUSTERMAP = 'tool data/clustermap.png'
    cluster_grid.savefig(PATH_FOR_CLUSTERMAP)

    print('Clustermap saved to ' + PATH_FOR_CLUSTERMAP)





    print('Plotting')

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
    PATH_FOR_HEATMAP = './heatmap.png'
    plt.savefig(PATH_FOR_HEATMAP)

    print('Heatmap saved to ' + PATH_FOR_HEATMAP)