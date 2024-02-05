import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


def main():
    # get all folders in the current directory
    folders = [f for f in os.listdir('.') if os.path.isdir(f)]

    results_of_all_folders = []

    for folder in folders:
        subfolders = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]
        dict = {}
        for subfolder in subfolders:
            df = pd.read_csv(os.path.join(folder, subfolder, 'results.csv'), index_col=0)
            # only keep the first and last column
            df = df.iloc[:, [-1]]

            # remove the paranthesis and all the text inside it from all cells
            df = df.replace(to_replace='\(.*\)', value='', regex=True)
            # make into floats
            df = df.astype(float)
            # For each row make sure that the dictionary has a key for it using the name
            # of the row. Then add the value of the cell to the list of values for that key.
            for row in df.iterrows():
                if row[0] not in dict:
                    dict[row[0]] = []
                dict[row[0]].append(row[1][0])

        averages = {}
        for key in dict:
            averages[key] = np.mean(dict[key])

        stds = {}
        for key in dict:
            stds[key] = np.std(dict[key])

        # make a dataframe from the averages and stds
        df = pd.DataFrame.from_dict(averages, orient='index')
        df.columns = ['Average']
        df['Std'] = stds.values()

        df_only_ensemble = df.iloc[[1,3,5,7,9,11], :]

        # remove "ensamble" from the row names
        df_only_ensemble.index = df_only_ensemble.index.str.replace('_ensemble', '')

        # make a bar plot of the averages and stds
        df_only_ensemble.plot.bar(y='Average', yerr='Std', rot=0)
        # make the plot bigger
        plt.gcf().set_size_inches(5, 3)

        # remove the legend
        plt.legend().remove()
        # Color the bars
        #ax = plt.gca()
        # ax.patches[0].set_facecolor('r')
        # ax.patches[1].set_facecolor('g')
        # ax.patches[2].set_facecolor('b')
        # ax.patches[3].set_facecolor('y')
        # ax.patches[4].set_facecolor('c')
        # #ax.patches[5].set_facecolor('m')
        #ax.patches[6].set_facecolor('k')
        
        plt.title(folder)
        plt.tight_layout()
        plt.savefig(folder + '.png')
        plt.close()

        results_of_all_folders.append(df)

    # Combine all the dataframes into one
    df = pd.concat(results_of_all_folders, axis=1)
    column_names = []
    for folder in folders:
        column_names.append(folder + ' Average')
        column_names.append(folder + ' Std')
    
    df.columns = column_names
       

    # save the dataframe to a csv file
    df.to_csv('results.csv')



if __name__ == '__main__':
    main()