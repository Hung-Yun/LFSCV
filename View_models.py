import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 0)
df = pd.read_pickle('Log/EN.pkl')
print('Model index\tConc\tSigma\tSize\tDate')
for i in range(len(df)):
    print(f'Model {i}:\t{df.iloc[i].Conc}\t{df.iloc[i].Sigma}\t{df.iloc[i].Size}\t{df.iloc[i].Date}')

# TODO: allow viewing involved sessions and data distribution.
# TODO: add logger
