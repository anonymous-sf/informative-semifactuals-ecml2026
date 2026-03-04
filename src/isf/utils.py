import numpy as np
import pandas as pd

def load_data(data_desc, target):
    data = pd.read_csv('../../data/'+data_desc+'_sc.csv')
    y = np.array(data[target])
    df_sub = data.loc[:, data.columns != target]
    x = np.array(df_sub)
    cols = list(df_sub.columns)

    # Calculate the  bounds of the dataset
    min_vals = x.min(axis=0)
    max_vals = x.max(axis=0)

    return x, y, min_vals, max_vals, df_sub, cols