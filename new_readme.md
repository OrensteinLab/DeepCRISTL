
# DeepCRISTL

A tool to predict CRISPR/Cas9 gRNA on-target efficiency for specific cellular contexts utilizing transfer-learning. The repository for the paper [DeepCRISTL: deep transfer learning to predict CRISPR/Cas9 functional and endogenous on-target editing efficiency](https://academic.oup.com/bioinformatics/article/38/Supplement_1/i161/6617528)

## Requirements

Our tool has been tested on the following configuration on a linux machine:
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
 - CRISPRoff : 1.1.2
 - ViennaRNA 2.5.0a5
 



## Running the Tool

### Checking Vvailable Models
To list available models, use the following command:
```sh
python tool.py --action list
```
Note that the model `no_transfer_learning` is always available

### Prediction
#### Example input file: `example.csv`

| 30mer                         |
|--------------------------------|
|CCTCGTTGCTATCTACCACAAGCAGGGGCG|
|CAACTTCTTCTCAGTTCAAGTAACAGGTAA|
|GGAATTCCGGGCATCCCTGTACAAGGGCGT|
|GAACTCGGCATTCGAGCGAAACTGGGGCTG|
|CACGACCAGTGCCCAAAACAGCTTAGGAGA|
|GACTACATGAACATGACTCCCCGGAGGCCT|
|CAATGACACTCAGGCTGCTGTTCTTGGCTC|
|CGTCGCTGTTCACGCCCTTGTACAGGGATG|
|ATTGCTCCTCTCGTTGTCTAGGTAAGGCGG|
|CCTCCGCCTTACCTAGACAACGAGAGGAGC|
|CCTCTCGTTGTCTAGGTAAGGCGGAGGGTA|
|GCCCTCATCAGAACAATGACACTCAGGCTG|
|TTTGTTATGGCTTGCTAGTGACAGTGGCTC|
|CTGGATAGGGGTCCCTGTCAGGGGCGGTAC|
|TGTACAAGGGCGTGAACAGCGACGTGGAAG|


#### How to predict
The input file should be located at `CRISPRon/tool data/input` and should be a CSV file containing only 1 column `30mer`. For a specific input file `X.csv` and a model `M`, use the following command:

```sh
python tool.py --action prediction --input_file X --model_to_use M
```

The prediction results will appear in `CRISPRon/tool data/output`

### Fine-Tuning on New Data
#### Example Target File: `example.csv`

| 30mer                        | mean_eff          |
|------------------------------|-------------------|
| AGGAAGCGTACCCCCAGGTCTTGCAGGTCC | 0.0058362780473332 |
| TTTTCCAATTGCCTTCAGATCAATAGGCTT | 0.0               |
| CCTTACAGGGCGCTCCATATTCGCAGGTGC | 0.1106926468458475 |
| TGTCCTCGTCCTCCAGCTGTTATCTGGAAG | 0.0094791555458701 |
| ACCTTCTCAATTAAATCTGACGTCTGGGGT | 0.0986944806386289 |
| ACCATCCGCCTGCGAGGCACGTAACGGAGC | 0.0050379567194411 |
| TCGGCATGATTGCCAACTCCGTGGTGGTCT | 0.0788925346952764 |
| ATAGTTTCTTTGGTCCCACGCCTGCGGCAC | 0.057214890077075  |
| CTTACCCACTACTATGATGATGCCCGGACC | 0.1051214533208349 |
| ATGTGCGCACCTGCATCCCCAAAGTGGAGC | 0.0745012100079162 |
| TTCCTGTAGGAGCACTGTCGACCCTGGCAT | 0.0805055090371068 |
| TGGCCAAGCCGTGGAGTGCTGCCAAGGGGA | 0.5331169658510118 |
| AACACGGTTCAACACCAGTTTGATTGGTTC | 0.0283029080994734 |
| GCTTGCGATGCCGGTACATCCAAAAGGCCA | 0.0448409515847471 |
| ... | ...

To fine-tune on new data, place the datasets CSV file in `CRISPRon/tool data/datasets`. The CSV file should contain 2 columns: `30mer` and `mean_eff`, with `mean_eff` normalized between 0 and 1. For a specific dataset file `D.csv`, use the following command:

`python tool.py --action new_data --new_data_path D`

Note that this will create a model with the same name as the fine-tune csv file.



## Acknowledgements

 - [CRISPRon](https://www.nature.com/articles/s41467-021-23576-0) for the architecture and pre-training data.
 - [Kim2019](https://www.science.org/doi/10.1126/sciadv.aax9249) for part of the pre-training data.
 - [DeepHF](https://www.nature.com/articles/s41467-019-12281-8) for models tested in the original paper.
 - [CRISPRoff](https://bulldogjob.com/news/449-how-to-write-a-good-readme-for-your-github-project) for the feature calculation used as part of the CRISPRon pipeline.
  - [Haussler](https://link.springer.com/article/10.1186/s13059-016-1012-2) for the target datasets.
- [Leenay](https://www.nature.com/articles/s41587-019-0203-2) for the target dataset.


## Contact
In case of issues with the tool, you may contact us at yaron.orenstein@biu.ac.il