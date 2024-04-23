# Get data and remove leakage
[Insert instructions on how to get and where to place]

# Preprocessing
DeepHF
python main.py -s_type preprocessing
python main_tl.py -s_type preprocess--tl_data ALL_ORIGINAL_DATA --tl_data_category U6T7 

CRISPRon
python main.py -s_type preprocess --tl_data ALL_ORIGINAL_DATA --tl_data_category U6T7 


# DeepHF cross-v
for each config [wt, hf, esp, multi_task]
python main.py -s_type cross_v --enzyme hf
 
delete results/pretrain
delete models/pre_train/DeepHF_old/*

# DeepHF full-cross-v
python main.py -s_type cross_v --enzyme multi_task

# DeepHF Fine Tune
[EXPLAIN WHERE TO PLACE MODELS]
python main.py -s_type full_sim --tl_data ALL_ORIGINAL_DATA --tl_data_category U6T7 

# CRISPRon Fine Tune

# Tool


