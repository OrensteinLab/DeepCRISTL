from scripts_tool import configurations as cfg
from scripts_tool import preprocess
#from scripts_tool import data_handler as dh
#from scripts_tool import cross_validation_tl as cv_tl
#from scripts_tool import ensemble_util
#from scripts_tool import  hyper_parameter_search as hps
import keras

#interperter = ModelInterpertation

if __name__ == '__main__':
    # Get configs
    config = cfg.get_parser()

    if config.action == 'new_data':
        preprocess.prepare_inputs(config)
        exit()

    if config.action == 'prediction':
        exit()

    if config.action == 'heat_map':
        exit()

    if config.action == 'saliency_maps':
        exit()
