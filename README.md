
# DeepCRISTL

A tool to predict CRISPR/Cas9 gRNA on-target efficiency for specific cellular contexts utilizing transfer-learning. The repository for the paper [DeepCRISTL: deep transfer learning to predict CRISPR/Cas9 functional and endogenous on-target editing efficiency](https://academic.oup.com/bioinformatics/article/38/Supplement_1/i161/6617528)

## Requirements

Our tool has been tested on the following configuration on a Linux machine:
 - python 3.6.13
 - h5py 2.10.0
 - Keras 2.4.3
 - Levenshtein 0.21.1
 - logomaker 0.8
 - matplotlib 2.2.0
 - numpy 1.19.5
 - pandas 1.1.5
 - scikit-learn 0.20.0
 - scipy 1.5.4
 - seaborn 0.11.2
 - tensorflow 2.4.1
 - CRISPRoff 1.1.2
 - ViennaRNA 2.5.0a5
 



## Running the Tool

First, navigate to the CRISPRon folder:
```sh
cd CRISPRon
```

### Checking Available Models
To list available models, use the following command:
```sh
python tool.py --action list
```
Note that the model `no_transfer_learning` is always available.

### Prediction
The input file should be located at `CRISPRon/tool data/input` and should be a CSV file containing only one column `30mer`. An example file named `example.csv` is provided with the sequences of the dataset `doench2014-Hs`. To run the script, input the following command:

```sh
python tool.py --action prediction --input_file <file_name> --model_to_use <model_name>
```

An example run:
```sh
python tool.py --action prediction --input_file example --model_to_use leenay
```


Make sure that `<file_name>` doesn't contain the `.csv` extension. 

The prediction results will appear in `CRISPRon/tool data/output`

### Fine-Tuning on New Data
To fine-tune on new data, place the datasets CSV file in `CRISPRon/tool data/datasets`. The CSV file should contain two columns: `30mer` and `mean_eff`, with `mean_eff` normalized between 0 and 1. An example file named `example.csv` is provided with the sequences and labels for the dataset `doench2014-Hs`. To run the script, input the following command:

```sh
python tool.py --action new_data --new_data_path <dataset_name>
```

An example run:
```sh
python tool.py --action new_data --new_data_path example
```


Make sure that `<dataset_name>` doesn't contain the `.csv` extension.

This command will create a model with the same name as the fine-tune CSV file.



## References

 - [CRISPRon](https://www.nature.com/articles/s41467-021-23576-0) for the architecture and pre-training data.
 - [Kim2019](https://www.science.org/doi/10.1126/sciadv.aax9249) for part of the pre-training data.
 - [DeepHF](https://www.nature.com/articles/s41467-019-12281-8) for models tested in the original paper.
 - [CRISPRoff](https://bulldogjob.com/news/449-how-to-write-a-good-readme-for-your-github-project) for the feature calculation used as part of the CRISPRon pipeline.
  - [Haussler](https://link.springer.com/article/10.1186/s13059-016-1012-2) for the target datasets.
- [Leenay](https://www.nature.com/articles/s41587-019-0203-2) for the target dataset.

## Licensing Information

- The **code** in this repository is licensed under the [MIT License](./LICENSE).
- The **models** in this repository were trained on external datasets. Their use may be subject to the licensing terms of the datasets from which they were derived. Please consult the following studies for information on the dataset licenses:

  ### Pretraining Datasets:
  - **CRISPRon pretrain datasets**:
    - [Enhancing CRISPR-Cas9 gRNA efficiency prediction by data integration and deep learning](https://www.nature.com/articles/s41467-021-23576-0)
    - [Prediction of the sequence-specific cleavage activity of Cas9 variants](https://www.nature.com/articles/s41587-020-0537-9)

  ### Fine-tuning Datasets:
  - **Compiled datasets from Haeussler's paper**:
    - [Evaluation of off-target and on-target scoring algorithms and integration into the guide RNA selection tool CRISPOR](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-016-1012-2)
  - **Leenay study**:
    - [Large dataset enables prediction of repair after CRISPR-Cas9 editing in primary T cells.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7388783/)



## Contact
In case of issues with the tool, you may contact us at yaron.orenstein@biu.ac.il.
