from rdkit import Chem
import numpy as np
import pandas as pd
import sys

smi_file = "derived/compounds_desalt.smi"

substr_list = [
    "[N;D3]C(*)=O", "*[O;D2]*", "*N(*)*", 'F', 'Cl', 'Br',
    '*[N;D2]*', 'OC', '*C([O;D1])=O', 'O=a', 'Oc', '*N(*)S(*)(=O)=O',
    'Nc', 'NC', '*OC(*)=O', 'C=C', '*C(*)=O', '*N(*)C(=O)N(*)*',
    '[S;D2]', 'C#N', '*OC(=O)[N;D3]', 'N1CCCCC1', 'N1CCNCC1', '*-!@*',
    'a-!@a', 'a-!@A', 'a-!@[R]', 's', '[R][O;D1]'
]

smiles = []
ind_names = []
smiles_dict = {}
sys.stderr.write("read molecules\n")
with open(smi_file, 'r') as iFile:
    for i, s in enumerate(iFile):
        S = s.strip().split()
        smiles.append([S[0], i])
        ind_names.append(S[1])
        smiles_dict[S[1]] = S[0]
sys.stderr.write(f"Number of uniq SMILES: {len(smiles_dict)},"
                 f" number of original compounds {len(smiles)}\n")

sys.stderr.write("convert molecules to mol object\n")
all_mols = []
for i in list(smiles_dict):
    m = Chem.MolFromSmiles(smiles_dict[i])
    m.SetProp("_Name",i)
    all_mols.append(m)

sys.stderr.write(f"calculate match to {len(substr_list)} fragments\n")
df = pd.DataFrame({'Id': list(smiles_dict)})
sub_mat = []
for i, smart in enumerate(substr_list):
    sub_id = f'sma_{i}'
    sub_col = []
    for m in all_mols:
            bis = m.GetSubstructMatches(Chem.MolFromSmarts(smart))
            sub_col.append(len(bis))
    df[sub_id] = np.array(sub_col)

df.to_csv('derived/substr_count.csv', index=False)
sys.stderr.write("done!\n")
