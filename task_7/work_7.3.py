import pandas as pd
import numpy as np

wells_na = pd.read_csv('wells_info_na.csv')
df_filled = wells_na.copy()

numeric_columns = ['LatWGS84', 'LonWGS84', 'PROP_PER_FOOT']
for col in numeric_columns:
    if col in df_filled.columns:
        df_filled[col] = df_filled[col].fillna(df_filled[col].median())

categorical_columns = ['CompletionDate', 'FirstProductionDate', 'formation', 'BasinName', 'StateName', 'CountyName']
for col in categorical_columns:
    if col in df_filled.columns:
        mode_values = df_filled[col].mode()
        df_filled[col] = df_filled[col].fillna(mode_values[0])

df_filled.to_csv('wells_info_na_filled.csv', index=False)