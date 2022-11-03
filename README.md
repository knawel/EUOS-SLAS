# EUOS-SLAS
Solubility prediction


# Data

All data is stored in data folder:
```
data/
├── derived
│   ├── compounds_desalt.smi
│   └── compounds.smi
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

## Workflow

All derivative data is sored in `derived` folder.

1. Extract smiles **compounds.smi**
2. De-salt (obabel -r -xc ...) **compounds_desalt.smi**
3. Calc descriptors (CDK, selected constitutional, topological, electronic) **cdk_descr.csv**
4. Calc fragments composition by **gen_substr.py**

The final data for ML/NN is in the `preprocessed` folder (pandas pickle files).
Data is raw (not normalized)
X.pk - features with names (**Id**)
Y.pk - labels
test.pk - data for prediction