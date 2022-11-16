import arff, numpy as np
import pandas as pd
import sys

# input arguments:
if len(sys.argv) != 3:
    sys.stderr.write('USAGE: *.py <input arff file> <output csv file>\n')
    sys.stderr.write('Convert BlueDesc output ARFF to the standard CSV file\n')
    exit()
for arg in sys.argv:
    if arg == '-h' or arg == '--help':
        sys.stderr.write('USAGE: *.py <input arff file> <output csv file>\n')
        sys.stderr.write('Convert BlueDesc output ARFF to the standard CSV file\n')
        exit()

in_file = sys.argv[1]
out_file = sys.argv[2]

dataset = arff.load(open(in_file))
data = np.array(dataset['data'],dtype=np.float64)
header = np.array(dataset['attributes'])[:, 0]

mols = []
with open(in_file, 'r') as iFile:
    for a in iFile:
        if a[:15] =='% NAME OF MOLEC':
            mol_id = a.strip().split(' ')[-1]
            mols.append(mol_id)

final_df = pd.DataFrame(data, columns=header)
final_df.index = mols
final_df.dropna(axis=1, inplace=True)
final_df.index.name = 'Id'
# Add mean values for Na3VO4
final_df.loc['EOS100042'] = final_df.mean()
sys.stdout.write(f"added mean values for EOS100042 (Na3VO4)\n")
final_df.to_csv(out_file)
sys.stdout.write(f"Shape: {final_df.shape}\n")

