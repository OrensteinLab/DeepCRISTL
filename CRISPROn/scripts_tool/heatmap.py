

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
    #models = []
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

    # # make into a dataframe
    # dataframe = pd.DataFrame(numpy_array, columns=datasets, index=models)


    # START OF EDIT ### TODO: Remove the addition of doench2016plx

    # doench2016_hg19_row_index = datasets.index('doench2016_hg19')
    # doench2016_hg19_column_index = models.index('doench2016_hg19')

    # # Duplicate the row corresponding to 'doench2016_hg19'
    # numpy_array = np.insert(numpy_array, doench2016_hg19_row_index+1, numpy_array[doench2016_hg19_row_index], axis=0)

    # # Update the datasets list to reflect the addition
    # datasets.insert(doench2016_hg19_row_index+1, 'doench2016plx_hg19')

    # # Duplicate the column corresponding to 'doench2016_hg19'
    # numpy_array = np.insert(numpy_array, doench2016_hg19_column_index+1, numpy_array[:, doench2016_hg19_column_index], axis=1)

    # transpose the numpy array to make it easier to work with
    numpy_array = numpy_array.T

    # print(numpy_array.shape)

    # # Update the models list to reflect the addition
    # models.insert(doench2016_hg19_column_index+1, 'doench2016plx_hg19')

    # Convert the updated NumPy array to a pandas DataFrame
    dataframe = pd.DataFrame(numpy_array, columns=models, index=datasets)
    
    # save dataframe to a csv file
    dataframe.to_csv('tool data/output/spearmans.csv')
    
    # END OF EDIT ###

    generate_one_heatmap(dataframe)





    
def get_reordered(numpy_array, datasets_order, models, datasets):
    # reorder the rows in the numpy array and the models
    new_numpy_array = numpy_array[datasets_order]
    new_datasets = [datasets[i] for i in datasets_order]


    #index_of_no_tl = models.index('no_transfer_learning')

    #no_tl_numpy_row = new_numpy_array[index_of_no_tl]

    # remove the row of no transfer learning
    #new_numpy_array = np.delete(new_numpy_array, index_of_no_tl, axis=0)

    model_order =[]
    for i in range(len(models)):
        if models[i] == 'no_transfer_learning':
            model_order.append(len(models)-1)
            continue
        index_of_model_in_datasets = datasets.index(models[i])
        model_order.append(datasets_order.index(index_of_model_in_datasets))

    

    


    



    # reorder the columns in the numpy array and the datasets
    new_numpy_array = new_numpy_array[:, model_order]
    new_models = [models[i] for i in model_order]

    return new_numpy_array, new_models, new_datasets


def get_clustermap_ordering(dataframe):

    
    # create a clustermap
    cluster_grid = sns.clustermap(dataframe, cmap='viridis', method='average', row_cluster=True, col_cluster=True)     
    row_order = cluster_grid.dendrogram_row.reordered_ind
    #print(row_order) # ordering of datasets

    return row_order





def generate_one_heatmap(dataframe):
    no_tl_col = dataframe['no_transfer_learning']

    df_only_targets = dataframe.drop(columns=['no_transfer_learning'])

    # sort the dataset rows based on alphabetical order
    df_only_targets = df_only_targets.sort_index()

    # sort the model columns based on alphabetical order
    df_only_targets = df_only_targets[sorted(df_only_targets.columns)]



    order = get_clustermap_ordering(df_only_targets)

    # reorder the dataframe rows
    reordered_df = df_only_targets.iloc[order]

    # reorder df columns based on order variable
    order_columns = order
    reordered_df = reordered_df[reordered_df.columns[order_columns]]


    # reorder no_tl_col based on the order variable
    no_tl_col = no_tl_col[order_columns]

    # add the no_tl_col to the reordered_df
    reordered_df['no_transfer_learning'] = no_tl_col

    print(reordered_df)
   
    




    models = reordered_df.columns
    datasets = reordered_df.index
    numpy_array = reordered_df.to_numpy()

    print(numpy_array.shape)




    # make a bigger plot
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)


    
    # Create a heatmap using Matplotlib's imshow function
    heatmap = ax.imshow(numpy_array, cmap='coolwarm', interpolation='nearest')


    # Add the values to the heatmap
    for i in range(len(datasets)):
        for j in range(len(models)):
            text = ax.text(j, i, format(numpy_array[i][j], '.2f'),
                        ha="center", va="center", color="black", fontsize=14)

    # Add colorbar to the right of the heatmap
    cbar = plt.colorbar(heatmap, ax=ax, fraction=0.01, pad=0.03, orientation='vertical')
    cbar.set_label('Spearman\'s R', rotation=270, labelpad=20, fontsize=14)



    #cbar_ax = fig.add_axes([0.05, 0.1, 0.4, 0.03])  # Adjust these values as needed
    #fig.colorbar(heatmap, cax=cbar_ax, orientation='horizontal')

    #cbar.ax.set_title('Spearman\'s R', pad=10, fontsize=18)

    # Set labels for the axes
    plt.xlabel('Model', fontsize=18, labelpad=20)
    plt.ylabel('Dataset', fontsize=18, labelpad=20)

    # Set the title for the heatmap
    #plt.title('Spearmans R', fontsize=20)

    # Set labels for each column base on the dataset
    plt.xticks([i for i in range(len(models))], models, rotation=90, fontsize=14)

    # Set labels for each row base on the model
    plt.yticks([i for i in range(len(datasets))], datasets, fontsize=14)

    # add serperation between the last column and the rest
    plt.axvline(x=len(models)-1.5, color='black', linewidth=2)

    # make the outside lines of the heatmap thicker
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)


    # adjust layout
    plt.tight_layout()

    # Save the heatmap to a file TODO: change it to output again
    #PATH_FOR_HEATMAP = 'tool data/output/heatmap.png'
    PATH_FOR_HEATMAP = 'tool data/output/'  + 'heatmap.png'
    plt.savefig(PATH_FOR_HEATMAP)

    print('Heatmap saved to ' + PATH_FOR_HEATMAP)

