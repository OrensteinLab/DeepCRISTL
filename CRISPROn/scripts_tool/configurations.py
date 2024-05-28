import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--action',
                         type=str,
                           default=None,
                           help= 'the simulation to perform.\n new_data requires new_data_path\n prediction requires input_file and model_to_use\n',
                           choices=['new_data', 'list', 'prediction', 'heat_map', 'saliency_maps', 'preprocess_pretrain_data'])

    parser.add_argument('--new_data_path',
                         type=str,
                           default=None,
                           help= 'name of the dataset file to be preprocessed and trained on')
    
    parser.add_argument('--input_file',
                         type=str,
                           default=None,
                           help="name of the file to be used for prediction, is used with conjunction with --model_to_use")
    
    parser.add_argument('--model_to_use',
                         type=str,
                           default=None,
                           help="name of the model to be used for prediction, is used with conjunction with --input_file")
    
    parser.add_argument('--recalculate_spearmans',
                        action='store_true',
                          help="Recalcualte spearmans if making a heatmap")
    


  


    config = parser.parse_args()

    
    # if no action is specified, the program will exit
    if config.action is None:
        print('No action specified >>>> exiting\nRun python tool.py -h for help on how to use the tool.')
        exit(1)
    
    if config.action == 'new_data':
        if config.new_data_path is None:
            print('No new data path specified >>>> exiting\nRun python tool.py -h for help on how to use the tool.')
            exit(1)

    if config.action == 'prediction':
        if config.input_file is None or config.model_to_use is None:
            print('No input file or model to use specified >>>> exiting\nRun python tool.py -h for help on how to use the tool.')
            exit(1)

    if config.action in ['new_data']:
      from keras.optimizers import RMSprop
      config.init_lr = 0.0043
      config.optimizer = RMSprop
      config.batch_size = 80

    


    return config