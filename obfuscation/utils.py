import pandas as pd
import numpy as np

def sigma_max(df:pd.DataFrame, col:str, is_standard:bool=True):
    return 1 if is_standard else df[col].max()

def choose_from_unselected(item_set, beta, n_rated):
    try:
        return np.random.choice(item_set, int(beta*n_rated), replace=False)
    except ValueError:
        return item_set

    
# TODO deal with a list of target cols instead of one at a time
def transform_standardize(df:pd.DataFrame, id_col:str, target_col:str):
    df = df.copy()
    df['z_'+target_col] = df.groupby(id_col)[target_col].transform(lambda x: (x - x.mean())/x.std())
    return df

# TODO deal with a set of target cols; in this case it might be more efficient to create a separate df in which
# for each user we store 2 cols for each target col, corresponding to the mean and std in the original var
# ex. rating --> rating_std, rating_mean
def transform_compute_desc_stats(df:pd.DataFrame, id_col:str, target_col:str):
    df = df.copy()
    df['user_{}_std'.format(target_col)] = df.groupby(id_col)[target_col].transform(lambda x: x.std())
    df['user_{}_mean'.format(target_col)] = df.groupby(id_col)[target_col].transform(lambda x: x.mean())
    return df

def transform_reverse_standardisation(df:pd.DataFrame, target_col:str):
    df = df.copy()
    df['{}_noise'.format(target_col)] = df['z_{}_noise'.format(target_col)] * df['user_{}_std'.format(target_col)] + df['user_{}_mean'.format(target_col)]
    return df