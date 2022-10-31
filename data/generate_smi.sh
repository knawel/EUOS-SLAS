awk 'NR>1' train.csv | cut -f 1,2 -d ',' | sed -r 's/^([^,]+),(.+)$/\2\t\1/' | tee train.smi | wc -l
awk 'NR>1' test.csv | cut -f 1,2 -d ',' | sed -r 's/^([^,]+),(.+)$/\2\t\1/' | tee test.smi | wc -l
