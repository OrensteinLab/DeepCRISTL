import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


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
            data_paths = ['leenay','chari2015Train293T','doench2014-Hs','doench2014-Mm','doench2016_hg19','hart2016-Hct1162lib1Avg','hart2016-HelaLib1Avg','hart2016-HelaLib2Avg','hart2016-Rpe1Avg','morenoMateos2015','xu2015TrainHl60','xu2015TrainKbm7']
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

        all_available_models = utils.get_all_models()
        if config.model_to_use not in all_available_models:
            print(f'Model {config.model_to_use} not found')
            print(f'Available models:')
            max_width = len(str(len(all_available_models))) + 2
            for i, folder in enumerate(all_available_models):
                print(f'{i+1: <{max_width}} {folder}')
            exit()
        else:
            print(f'Loading models')
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
        from scripts_tool import models_util
        from scripts_tool import testing_util
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt


        models_and_datasets = utils.get_all_models()
        models_spearmans = {}
        print('Calculating spearmans')
        for model in models_and_datasets:
            print(f'\n\n\nCalculating spearmans for model trained on {model}')
            print('-----------------------------------\n\n')
            config.model_to_use = model
            models = models_util.load_tl_models(config)
            models_spearmans[model] = {}
            for dataset in models_and_datasets:
                print(f'Calculating spearmans on {dataset}')
                DataHandler = dh.get_data_from_dataset(dataset)
                spearmanr, pvalue = testing_util.get_spearmanr(models, DataHandler) #TODO: add p value to the output?
                models_spearmans[model][dataset] = spearmanr

        print('Spearmans:')
        for model in models_spearmans:
            print(f'\n\n\n{model}')
            print('-----------------------------------\n\n')
            for dataset in models_spearmans[model]:
                print(f'{dataset}: {models_spearmans[model][dataset]}')


        # make into a 2d array and plot
        print('Plotting')
        numpy_array = []
        for model in models_spearmans:
            numpy_array.append([models_spearmans[model][dataset] for dataset in models_spearmans[model]])
        
        # Create a heatmap using Matplotlib's imshow function
        heatmap = plt.imshow(numpy_array, cmap='viridis', interpolation='nearest')

        # Add colorbar to the right of the heatmap
        plt.colorbar()

        # Set labels for the axes
        plt.xlabel('Column')
        plt.ylabel('Row')

        # Set the title for the heatmap
        plt.title('Heatmap Title')

        # Set labels for each column base on the dataset
        plt.xticks([i for i in range(len(models_spearmans))], models_spearmans.keys())

        # Set labels for each row base on the model
        plt.yticks([i for i in range(len(models_spearmans))], models_spearmans.keys())

        # Save the heatmap to a file
        plt.savefig('heatmap.png')

      


        

    if config.action == 'saliency_maps':
        exit()
