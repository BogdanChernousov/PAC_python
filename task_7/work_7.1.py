import pandas as pd
import numpy as np

np.random.seed()
df = pd.DataFrame(np.random.random((10, 5)), columns=['A', 'B', 'C', 'D', 'E'])
print(df)

def mean_above_03(row):
    filtered = [x for x in row if x > 0.3]
    return np.mean(filtered) if filtered else 0

df['mean_above_03'] = df.apply(mean_above_03, axis=1)
print(df)
