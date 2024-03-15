# Important
Use enviroment 1 for DeepHF models and enviroment 2 for CRISPRon models.
Follow the guide in this exact order.

# DeepHF 
1. Download supplementary data 2 from https://www.nature.com/articles/s41467-019-12281-8#Sec40
2. Remove the empty row and column and save it as "supplementary2.csv"
3. put it in data/main_dataframes/
4. run python 3.6 -u ./main.py -s_type preprocess
5. go to data/pre_train/DeepHF_old and save the csv files somewhere else -> those are the original pretrain files
6.Take the original pretrain files and put them in /new version
7. run python3.6 ./redistrubute_split.py 
8. Delete all the .pkl files in data/pre_train/DeepHF_old
9. Take the files from the output folder and put them in go to data/pre_train/DeepHF_old
10. Run python3.6 -u ./main.py -s_type preprocess 


# Leenay and U6T7 For DeepHF

1. Download leenay_full_data.csv from XXXXXXXXXXXXX
2. Download the supplementary table from CrisprON - a file named "13059_2016_1012_MOESM14_ESM.tsv" https://springernature.figshare.com/articles/dataset/Additional_file_14_of_Evaluation_of_off-target_and_on-target_scoring_algorithms_and_integration_into_the_guide_RNA_selection_tool_CRISPOR/4466045
3. put both files in data/main_dataframes/
4. Download the fix files from github - all the .tab files in https://github.com/maximilianh/crisporPaper/tree/master/effData
5. Put them in data/main/dataframes/CrisprOR fixes
6. run python3.6 -u fix_supp.py
7. run python 3.6 -u ./main_tl.py -s_type preprocess --tl_data leenay --tl_data_category leenay 
8. run python 3.6 -u ./main_tl.py -s_type preprocess --tl_data ALL_U6T7_DATA --tl_data_category U6T7 


# Leenay and U6T7 For DeepHF for CRISPRon
1. Go to the CRISPROn folder
2. run python 3.6 -u ./main.py -s_type preprocess --tl_data leenay --tl_data_category leenay 
3. run python 3.6 -u ./main.py -s_type preprocess --tl_data ALL_U6T7_DATA --tl_data_category U6T7 

