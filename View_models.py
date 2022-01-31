import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 0)
df = pd.read_pickle('Log/EN.pkl')
print(df.head().to_string())
