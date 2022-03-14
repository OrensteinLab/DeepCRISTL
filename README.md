# System requirements
* python=3.8
* numpy=1.14.0
* scipy=1.0.0
* h5py=2.7.1
* tensorflow=1.8.0
* keras=2.1.6
* scikit-learn=0.19.1
* biopython=1.71
* viennarna=2.4.5
* cutadapt=1.18
* matplotlib
* DotMap
* GPyOpt
* pandas

We have used the code of the DeepHF model as our baeline code.
#  Pre-train model
For training the pre-train model, use the file - "main.py".
* -s_type = the simulation type:
  * preprocess - preparing the data in objects and divide to train and test
  * train - training the model
  * postprocess - geting plots of training results
* --enzyme = the enzyme data that will be used as pre-train data - [wt, esp, hf, multi_task]
* Example code:
-s_type train --pre_train_data DeepHF_old   --enzyme multi_task

# Transfer learning model
For training the model on the endogenous and functional data, use the file - "main_tl.py"
* -s_type = the simulation type:
  * preprocess - prepare data
  * full_sim - run all transfer learning approaches
  * postprocess - recive plots of results

* --tl_data = The data which will be fine tuned on
*  --tl_data_category = The type of fine tuned data - [u6t7, leenay]
* Example code:
-s_type postprocess --tl_data doench2014-Hs --pre_train_data DeepHF_old  --tl_data_category u6t7 --enzyme wt


# Transfer learning with CRISPROn model
run the "CRISPRon/main.py" file with the following configurations:

Start from preprocessing the data using the followin code: 
-s_type preprocess --tl_data_category U6T7 --tl_data chari2015Train293T (choose the relevant expirement)

Then you can run the full simulation with:
-s_type full_Sim --tl_data_category U6T7 --tl_data chari2015Train293T

For reciving the finall results, run the postprocess:
-s_type postprocess --tl_data_category U6T7 --tl_data chari2015Train293T

# Model interpertation
For interperting the expirements, run the file "ModelInterpertations/model_interpertation.py". 
For any new expirement, add the expirement name to the expirement list in line 112. 
You can choose the pre train model with the "--enzyme" option. FOr the multi task model, write multi_task as in the example.
* Example code:
-s_type interpertation --enzyme multi_task --pre_train_data DeepHF_old

# Data:
All the data are publicly available at the following papers:
* DeepHF - https://www.nature.com/articles/s41467-019-12281-8
* Leenay - https://www.nature.com/articles/s41587-019-0203-2
* Haussler - https://link.springer.com/article/10.1186/s13059-016-1012-2
* CISPROn - https://www.nature.com/articles/s41467-021-23576-0

  
