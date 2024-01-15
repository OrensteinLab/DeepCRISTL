# Pretrain cross validation (with test)
python 3.6 -u ./main.py -s_type cross_v --enzyme wt

# Full cross validation (pretrain for tl)
python 3.6 -u ./main.py -s_type full_cross_v --enzyme wt

# TL all experiments
pythom 3.6 -u ./main.tl -s_type full_sim --tl_data xu2015TrainHl60 --tl_data_category U6T7 --pre_train_data DeepHF_old