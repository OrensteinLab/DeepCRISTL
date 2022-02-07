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
* matplotlib
* DotMap
* GPyOpt
* pandas

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

  
