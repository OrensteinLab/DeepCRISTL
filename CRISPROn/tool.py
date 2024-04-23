import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
import pandas as pd

from scripts_tool import configurations as cfg
from scripts_tool import data_handler as dh
from scripts_tool import utils
#from scripts_tool import ensemble_util
#from scripts_tool import  hyper_parameter_search as hps





if __name__ == '__main__':
    # Get configs
    config = cfg.get_parser()

    if config.action == 'new_data':
        import keras
        from scripts_tool import preprocess
        from scripts_tool import cross_validation_tl as cv_tl

        if config.new_data_path == 'ALL_ORIGINAL_DATA':
            data_paths = ['leenay','chari2015Train293T','doench2014-Hs','doench2014-Mm','doench2016_hg19', 'doench2016plx_hg19','hart2016-Hct1162lib1Avg','hart2016-HelaLib1Avg','hart2016-HelaLib2Avg','hart2016-Rpe1Avg','morenoMateos2015','xu2015TrainHl60','xu2015TrainKbm7']
        else:
            data_paths = [config.new_data_path]
        for data_path in data_paths:
            config.new_data_path = data_path
            print('Preprocessing new data - ', config.new_data_path)
            preprocess.prepare_inputs(config)

            print(f'Training on {config.new_data_path} dataset')
            train_types = ['LL_tl', 'gl_tl']

            DataHandler = dh.get_data(config)

            for train_type in train_types:
                print(f'#################### Running {train_type} model #############################')
                config.train_type = train_type
                config.save_model = False
                print(f'Running cross_v_HPS with {train_type} model')
                config.epochs = 100
                opt_epochs = cv_tl.cross_v_HPS(config, DataHandler)
                config.epochs = opt_epochs

                print(f'Running full_train with {train_type} model')
                config.save_model = True
                mean = cv_tl.train_6(config, DataHandler)

                keras.backend.clear_session()
            
            print('Training complete for ', config.new_data_path)


        
    if config.action == 'list':
        models = utils.get_all_models()
        print("Available models:")
        max_width = len(str(len(models))) + 2
        for i, folder in enumerate(models):
            print(f'{i+1: <{max_width}} {folder}')
        
    


    if config.action == 'prediction':
        from scripts_tool import models_util
        from scripts_tool import preprocess
        from scripts_tool import prediction

        all_available_models = utils.get_all_models() + ['no_transfer_learning']
        if config.model_to_use not in all_available_models:
            print(f'Model {config.model_to_use} not found')
            print(f'Available models:')
            max_width = len(str(len(all_available_models))) + 2
            for i, folder in enumerate(all_available_models):
                print(f'{i+1: <{max_width}} {folder}')
            exit()
        else:
            print(f'Loading models')
            if config.model_to_use == 'no_transfer_learning':
                models = models_util.load_no_tl_models()
            else:
                models = models_util.load_tl_models(config)

            print(f'Preparing data')
            sequences = preprocess.prepare_user_input(config)
            DataHandler = dh.get_user_input_data(sequences)

            print(f'Predicting')
            final_pred = prediction.predict(models, DataHandler)

            print(f'Saving prediction file')
            prediction.save_prediction_file(config, final_pred)

            print('Prediction complete')
            

    if config.action == 'heat_map':
        from scripts_tool import heatmap
        heatmap.generate_heatmap(config, from_pickle=(not config.recalculate_spearmans))
        

      
    if config.action == 'preprocess_pretrain_data':
        from scripts_tool import preprocess
        preprocess.prepare_pretrain_data(config)

        

    if config.action == 'saliency_maps':
        import tensorflow as tf


        from scripts_tool import models_util
        models = utils.get_all_models() + ['no transfer learning']
        #models = utils.get_all_models()
        datasets = utils.get_all_datasets()

        for model in models:
            saliency_dfs = []
            print(f'\n\n\nCalculating saliency maps for model trained on {model}')
            if model == 'no transfer learning':
                ensemble_models = models_util.load_no_tl_models()
            else:
                config.model_to_use = model
                ensemble_models = models_util.load_tl_models(config)

            for used_model in ensemble_models:

                DataHandler = dh.get_data_from_dataset(model)
                x = DataHandler['X_test']
                dg = DataHandler['dg_test']
                combined_data = [x, dg]
                # make into a tensor
                combined_data_tensors = [tf.convert_to_tensor(x) for x in combined_data]

            # Use a persistent GradientTape
                with tf.GradientTape(persistent=True) as tape:
                    # Watch each input tensor
                    for data_tensor in combined_data_tensors:
                        tape.watch(data_tensor)
                    
                    # Assuming your model takes a list of tensors as input
                    predictions = used_model(combined_data_tensors)
                    
                    # Compute a scalar loss
                    loss = tf.reduce_mean(predictions)

                # Compute gradients with respect to each input tensor
                gradients_x = tape.gradient(loss, combined_data_tensors[0])
                gradients_dg = tape.gradient(loss, combined_data_tensors[1])

                # Free up resources used by the persistent tape
                del tape

                # average the gradients for each sequence
                gradients_x = np.sum(gradients_x, axis=0) # TODO: make sure same results
                gradients_dg = np.sum(gradients_dg)

                 

                saliency_map = gradients_x 
                
                # print it with only 2 digits after tghe decimal point
                for i in range(saliency_map.shape[0]):
                    print(f'{i}:', end='\t')
                    for index, letter in enumerate(['A', 'C', 'G', 'T']):
                        value = saliency_map[i][index]
                        if value < 0:
                            print(f'{letter}: {saliency_map[i][index]:.2f}', end='\t')
                        else:
                            print(f'{letter}: {saliency_map[i][index]:.2f} ', end='\t')
                    print()

                saliency_df = pd.DataFrame(saliency_map, columns=['A', 'C', 'G', 'T'])
                saliency_dfs.append(saliency_df)

            # average the saliency maps
            saliency_df = pd.concat(saliency_dfs).groupby(level=0).mean()
                
            # make all values to 0 in the PAM's GG region
            saliency_df.loc[25] = 0
            saliency_df.loc[26] = 0

            import logomaker 
            # Create the logo
            logo = logomaker.Logo(saliency_df, center_values=False)

            # Style the logo as needed
            logo.style_xticks(rotation=90, fmt='%d', anchor=0) # Optional: style x-ticks if sequence positions are important
            labels = ['-4', '-3', '-2', '-1', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13',
                       '14', '15', '16', '17', '18', '19', '20', 'N', 'G', 'G', '24', '25','26']
            logo.ax.set_ylabel("Saliency (importance)") # Adjust label as needed
            logo.ax.set_xticklabels(labels, rotation=90)

            # Save the logo to a file in the tool data/output folder
            logo.fig.savefig(f'tool data/output/saliency_map_{model}.png', dpi=300, bbox_inches='tight')
            




 
            

            

            




