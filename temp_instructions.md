# Pretrain cross validation (with test)
python 3.6 -u ./main.py -s_type cross_v --enzyme wt


# How to get pretrain data
1. Download suplementry2.csv
2. put it in data/main_dataframes/
3. python 3.6 -u ./main.py -s_type preprocess
4. go to data/pre_train/DeepHF_old and save the csv files somewhere else -> those are the original pretrain files

# How to run a normal cross validation
1. Take the original pretrain files and put them in /new version
2. run python3.6 ./redistrubute_split.py
3. Take the files from the output folder and put them in go to data/pre_train/DeepHF_old
4. Delete all the .pkl files in data/pre_train/DeepHF_old
5. Run python3.6 -u ./main.py -s_type preprocess 
6. run for example python 3.6 -u ./main.py -s_type cross_v --enzyme wt


# How to run a full cross validation - pretrain for tl
1. Take the original pretrain files and put them in /new version
2. run python3.6 ./redistrubute_split.py remove_tl_leakage
3. Take the files from the output folder and put them in go to data/pre_train/DeepHF_old
4. Delete all the .pkl files in data/pre_train/DeepHF_old
5. Run python3.6 -u ./main.py -s_type preprocess 
6. run for example python 3.6 -u ./main.py -s_type full_cross_v --enzyme wt

# How to get TL data
1. Download the supplementary table from CrisprON - a file named "13059_2016_1012_MOESM14_ESM.tsv" https://springernature.figshare.com/articles/dataset/Additional_file_14_of_Evaluation_of_off-target_and_on-target_scoring_algorithms_and_integration_into_the_guide_RNA_selection_tool_CRISPOR/4466045
2. Download the fix files from github - all the .tab files in https://github.com/maximilianh/crisporPaper/tree/master/effData
3. Put them in data/main/dataframes/CrisprOR fixes
4. run fix_supp.py
5. For each experiment run  python 3.6 -u ./main_tl.py -s_type preprocess --tl_data ALL_U6T7_DATA --tl_data_category U6T7 
Note that in our experiments we put everything in the U6T7 category for convinicence.



# TL all experiments
pythom 3.6 -u ./main.tl -s_type full_sim --tl_data xu2015TrainHl60 --tl_data_category U6T7 






