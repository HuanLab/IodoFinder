# IodoFinder
[![Generic badge](https://img.shields.io/badge/IodoFinder-ver_1.0-<COLOR>.svg)](https://github.com/Huanlab/IodoFinder)
![Maintainer](https://img.shields.io/badge/maintainer-Tingting_Zhao-blue)

`IodoFinder` is a python script to recognize and clean iodinated compounds in LC-MS/MS analysis. 
From the feature table and raw LC-MS/MS data, it can automatically recognize the iodinated compounds,
and further flag the isotopes and adducts. Two machine learning models are provided for positive and negative ionization modes separately.

The program is written in the language Python and its source code is publicly available at [IodoFinder](https://github.com/Huanlab/IodoFinder).

<!-- TOC -->
* [IodoFinder](#iodofinder)
  * [Installation instructions](#installation-instructions)
    * [System requirements](#system-requirements-)
    * [IodoFinder installation](#iodofinder-installation)
    * [Download machine learning models](#download-machine-learning-model)
  * [Instructions for usage](#instructions-for-usage)
    * [Data preparation preparation](#data-preparation-preparation)
    * [Configure parameters](#configure-working-directory-and-set-parameters)
    * [Part 1 Extraciton of chlorinated compounds](#part-1-extraciton-of-chlorinated-compounds)
    * [Part 2 Removal of isotopes and adducts](#part-2-alignment-across-samples)
  * [Citation](#citation)
<!-- TOC -->


## Installation instructions
### System requirements 
Python version 3.9.19 or above is required. To run IodoFinder successfully, please install the following packages first:

```angular2html
* pandas 2.2.2
* numpy 2.0.0
* pyteomics 4.7.2
* joblib 1.4.2
```

### IodoFinder installation
Download IodoFinder script (IodoFinder.py) from [IodoFinder](https://github.com/Huanlab/IodoFinder)

### Download machine learning model
`Positive model` and `Negative model` can be freely downloaded in [machine learning models](https://github.com/HuanLab/IodoFinder/tree/main/machine_learning_models)

`Demo data` can be freely downloaded in [demo data](https://github.com/TingtingZhao81/IodoFinder/main/demo_data)

## Instructions for usage

IodoFinder contains two modules: <br>
[1. Extration of iodinated compounds](#Part-1-Identification-of-chlorinated-compounds) <br>
[2. Removal of false positives](#Part-2-Alignment-across-samples) <br>

### Data preparation preparation
1. Put convert raw lc-ms/ms data into mzML or mzXML format
2. Prepare the feature table in advance. <br>
   The format for customized feature table:<br>
- 'featureID': ID of the features
- 'mz': m/z of the features
- 'rt': retention time in seconds
- 'Int_S1', 'Int_S2': peak intensity (height/area) from each raw lcms file. The orders of these intenisty columns should match of the corresponding raw lcms file

| featureID | mz        | rt      | Int_S1 | Int_S2  |
|-----------|-----------|---------|--------|---------|
| 1         | 327.0745  | 1813.34 | 100000 | 500     |
| 2         | 274.2744  | 1821.80 | 20000  | 700000  |

### Configure working directory and set parameters
1. Specify the folder and format of raw lcms data in line 9 and 10
  ```angular2html
  directory = 'C:/Users/User/Tingting/2024-04-17-iodine compounds/Model_training/negative' 
  pattern = '*.mzML'
  ```
2. Specify the path of feature table in line 11
  ```angular2html
  ft_path = "C:/Users/User/Tingting/2024-05-24-Suwannee_HA_negative_mode/2024-05-23-HA/HA_I/extract_I_in_MS2/I_neg_3MB.csv"
  ```
3. Specify the path of machine learning model in line 12
  ```angular2html
  ft_path = model_path = 'C:/Users/User/Tingting/positive_model_on_all_train.pkl' # pos: 'positive_model_on_all_train.pkl' ; neg: 'negative_model_on_all_train.pkl'
  ```
4. Set parameters associated with LC-MS/MS data collection line 13-16
  ```angular2html
  ionization_model = 'P' # positive: 'P'; negative: 'N'
  rt_tol = 0.2 # 0.2 minutes rt tolerance to find the ms2 spectra for corresponding features
  mz_tol = 10 # 10 ppm mass error to find the ms2 spectra for corresponding features
  frag_mz_tol = 0.01 # 0.01 Da to find typical fragment ion or neutral loss
  ```

### Run IodoFinder
Click '**run**' button, two tables will be output:

1. Output the table of iodinated features
  ```
  # positive mode
  tb.to_csv(f"positive mode {len(tb)} features, {len(tb[tb['ms2_mz'] != ''])} has ms2 spectra {len(tb[tb['prediction'] == 1])} I compounds.csv")
  # negative mode
  tb.to_csv(f"negative mode {len(tb)} features, {len(tb[tb['ms2_mz'] != ''])} has ms2 spectra {len(tb[tb['I_frag_mz'] != 0])} I frag {len(tb[tb['prediction'] == 1])} I compounds.csv")
  ```
2. Output the table flagging the isotopes and adducts
  ```
  tb_sorted.to_csv(f'{len(tb_sorted)} iodine chemicals {isotope_num} isotopes {adduct_num} adducts.csv')
  ```

## Citation
If you use IodoFinder in your research, please cite the following paper:
 


