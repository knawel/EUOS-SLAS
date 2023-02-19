# Solubility of the small molecules

This project was created during participation in the Kaggle competition
([1st EUOS/SLAS Joint Challenge: Compound Solubility](https://www.kaggle.com/competitions/euos-slas/rules)).
There are several model, the best one is 27th place in the leaderboard.

# Abstract

The solubility is quite challenging parameter for prediction, a lot of factors influence on it: hydrophilicity, polarity, dipol moment, 
flexibility, charges, 3D shape of the molecule. So, it seems important to give to the model these pre-computed parameters. 
Moreover, in the medicinal chemistry there are known fragments/groups which usually increase / decrease solubility.
I included information about presence or absence of the such groups.

So, the dataset contains the following parameters: 
- parameters for phisics-based modeling (descriptors including 3D, obtained from protonated 3D conformers).
- occurrence of the knowledge-based chemical groups

This dataset was used in the different models: sklearn classifiers, pytorch, tensorflow deep neural networks. 

# Methods and data

## Dataset

The input data contains chemical structures (SMILES format) and solubility (three classes: 0 (low), 1 (medium), 2 (high)).
 The data is highly unbalanced: 93% of compounds belong to the high-soluble class.


## Workflow

All derivative data is stored in `derived` folder.

1. Extract smiles **compounds.smi**
2. De-salt (obabel -r -xc ...) **compounds_desalt.smi**
3. Calculate fragments composition by **gen_substr.py**
4. Generate 3D structures by ballon (3d), protonate by open babel at pH=7 (3dH)
5. Generate descriptors (CDK and BleDesc)
6. Collect the data using *preprocess.py*

The final pre-computed parameters are in the **preprocessed** folder.

## Pre-computed parameters

There are 203 parameter (see the names in [file](docs/parameters.txt))
- 151 physico-chemical descriptors ()
- 52 chemical fragments 

 
## Files with data

All data is stored in data folder:

```
data/
├── generate_smi.sh
├── preprocessed
│   ├── test.pk
│   ├── X.pk
│   └── Y.pk
├── preprocess.py
└── raw
    ├── 50_most_popular_groups.txt
    ├── example_predictors.csv
    ├── submission_template_rdm.csv
    ├── test.csv
    └── train.csv
```
Add data file should contain 'Id' field with IDs of the molecules.

The final data for ML/NN is in the `preprocessed` folder (pandas pickle files).
Data is raw (not normalized)
X.pk - features with names (**Id**)
Y.pk - labels
test.pk - data for prediction

## Trial of descriptors calculators 
I tried: BlueDesc, CDK, RDkit.

The protonation state seems to be important. I checked it: compared 3D structures after ballon and babel (pH7)
BlueDesc: (many descriptors) including TPSA, logP, HDB.
CDK Desc: the same. 
RDKit: it is not possible to load the molecules. 

So, CDK or BlueDesc descriptors for protonated molecules seems as the reasonable choice
CDK:
About CDK Length Over Breadth is not calculated for many cases.
Topological ind are slow (all Chi..., Vertex and Weighted path descriptors),
so I excluded these 6 descriptors from calculation (+ Length Over Breadth).

BlueDesc:
Failed to calc for EOS100042 (VO4), so manually added for it the mean values.  

## Feature selection

For each calculated parameters I calculated
1) p-value between each groups (group 0 vs group 1, ...)
2) most important features for classification

### Fragments


| descr  | SMARTS                | 0-1    | 0-2    | 1-2    |
|--------|-----------------------|--------|--------|--------|
| sma_1  | `*[O;D2]*`            | 0.011  | 1.144  | -1.323 |
| sma_13 | `NC`                  | -0.029 | -2.906 | 3.365  |
| sma_27 | `s`                   | 0.119  | 0.136  | 0.024  |
| sma_33 | `[#16][#6;R][#6;R]~O` | 0.027  | -0.229 | 0.310  |
| sma_38 | `[R]S(=O)(=O)[#6]`    | -0.200 | 0.007  | -0.310 |
| sma_43 | `N*(=O)N`             | 0.014  | 0.029  | -0.012 |


Important features for different classifiers:
RandomForestClassifier (> 0.05):  ['sma_13', 'sma_23', 'sma_25']
DecisionTreeClassifier (> 0.05): ['sma_13', 'sma_23', 'sma_25', 'sma_28']
sma_13 is `NC`, sma_23 is `*-!@*`, sma_25 is `a-!@A`, sma_28 is `[R][O;D1]`

### BlueDesc
p-val < 0.05

| descr             | 0-1    | 0-2    | 1-2    |
|:------------------|--------|--------|--------|
| ..count.NumberOfS | 0.315  | 0.048  | 0.426  |
| SCH-3             | -0.044 | -0.918 | 1.012  |
| SCH-5             | 0.020  | -2.264 | 2.683  |
| MOMI-XY           | -0.042 | 0.831  | -1.038 |
| Weta3.unity       | -1.107 | 0.019  | -1.737 |
| WD.unity          | 0.018  | -0.945 | 1.138  |
| ..count.SO2Groups | -0.027 | -0.579 | 0.640  |

Important features for different classifiers:
RandomForestClassifier (> 0.01):  ['ATSc2', 'ATSc3', 'ATSc4', 'ATSc5', 'Weta1.unity', 'Weta2.unity', 'WD.unity', 'RPCS', 'XLogP']
DecisionTreeClassifier (> 0.02): ['..count.AromaticBonds', 'WTPT-1', 'WTPT-2', 'chi1C', 'nAtomP', 'ATSc2', 'ATSc3', 'ATSc4', 'ATSc5', 'MOMI-Z',
       'Weta1.unity', 'Weta2.unity', '..PolarSurfaceArea', 'PPSA-3', 'RNCG', 'TPSA']

### CDK
p-val < 0.05

| descr       | 0-1    | 0-2    | 1-2    |
|-------------|--------|--------|--------|
| Weta3.unity | -1.108 | 0.019  | -1.738 |
| FMF         | -1.704 | 0.008  | -2.794 |
| khs.sCH3    | 1.772  | 2.326  | 0.042  |
| khs.tCH     | 0.031  | 2.538  | -2.897 |
| khs.ssNH2   | -0.351 | -0.455 | 0.006  |
| khs.aaN     | -0.726 | 0.045  | -1.174 |
| khs.ssO     | 0.011  | 1.144  | -1.323 |
| khs.aaS     | 0.119  | 0.136  | 0.024  |
| khs.ddssS   | -0.027 | -0.579 | 0.640  |
| MOMI-XY     | -0.042 | 0.832  | -1.038 |
| nRings5     | 0.007  | -1.062 | 1.257  |

RandomForestClassifier (> 0.015): ['ALogP', 'ALogp2', 'Wlambda1.unity', 'Wnu1.unity', 'Weta2.unity',
       'Weta3.unity', 'MOMI-XY', 'geomShape', 'XLogP']
DecisionTreeClassifier (> 0.02): ['ALogp2', 'Wlambda1.unity', 'nAromBond', 'XLogP']

### Conclusion

I selected substructures and CDK descriptors for predictions
