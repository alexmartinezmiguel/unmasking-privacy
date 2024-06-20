import pandas as pd
import numpy as np
import math
from io import StringIO
from .utils import *

def choose_from_unselected(item_set, beta, n_rated):
    try:
        return np.random.choice(item_set, int(beta*n_rated), replace=False)
    except ValueError:
        return item_set


def response_no_masking(df: pd.DataFrame, is_fixed_response:bool, id_col:str, interaction_col_names:list, n_groups:int=1, theta_max:float=1.):
    """
    Algorithms for obfuscating user-item interactions, without masking. 
    These correspond to Algorithms 5 and 6 in the original framework by Polat and Batmaz. 
    If fixed response, the same flipping threshold theta will be applied to the entire data frame. 
    If variable response, a different flipping threshold theta (not larger than theta_max) 
    is generated for each group of items, under each user.

    Args:
        df (pd.DataFrame): data frame to obfuscate
        is_fixed_response (bool): obfuscation type - fixed or variable response
        id_col (str): specify the id column in the data frame
        interaction_col_names (list): specify the cols that represent user-item interactions
        n_groups (int, optional): item groups per user. Defaults to 1.
        theta_max (float, optional): flipping threshold. Defaults to 1..
    """
    response_col_names = [col+'_r' for col in interaction_col_names]
    df = df.copy()
    df[response_col_names] = df[interaction_col_names].copy()
    user_groups = df.groupby(id_col)
    df['group'] = user_groups[id_col].transform(lambda x: np.random.choice(n_groups, len(x)))
    if is_fixed_response:
        # Algorithm 5
        # equivalent to flipping a coin with proba theta_max/2 - NOT REALLY! USE THE ONE BELOW!
        # this could be rewritten in a more concise way to cover both cases, but it would generate more redundant data in memory
#         df['r'] = df.groupby([id_col, 'group'])['group'].transform(lambda x: np.random.binomial(1, theta_max/2))
#         df.loc[df['r'] == 1, response_col_names] = 1 - df.loc[df['r'] == 1, response_col_names]
        df['r'] = df.groupby([id_col, 'group'])['group'].transform(lambda x: np.random.uniform())
        df['theta'] = theta_max
        df.loc[df['r'] >= df['theta'], response_col_names] = 1 - df.loc[df['r'] >= df['theta'], response_col_names]
    else:
        # Algorithm 6
        # for each group of each user
        # generate a random number between [0,1]
        df['r'] = df.groupby([id_col, 'group'])['group'].transform(lambda x: np.random.uniform())
        # and a theta betwen [0, theta_max]
        df['theta'] = df.groupby([id_col, 'group'])['group'].transform(lambda x: np.random.uniform(0, theta_max))
        df.loc[df['r'] >= df['theta'], response_col_names] = 1 - df.loc[df['r'] >= df['theta'], response_col_names]
    return df


def response_with_masking(df: pd.DataFrame, is_fixed_response:bool, id_col:str, item_col:str, interaction_col_names:list, n_groups:int=1, theta_max:int=1):
    """
    Algorithms for obfuscating user-item interactions, with masking. 
    This allows for concealling real user-item interactions, by generating user-item pairs based on
    those items that a user has not rated yet.
    These correspond to Algorithms 7 and 8 in the original framework by Polat and Batmaz. 
    If fixed response, the same flipping threshold theta will be applied to the entire data frame. 
    If variable response, a different flipping threshold theta (not larger than theta_max) 
    is generated for each group of items, under each user.


    Args:
        df (pd.DataFrame): data frame to be obfuscated
        is_fixed_response (bool): obfuscation type - fixed response or variable response
        id_col (str): specify the id column in the data frame
        item_col (str): specify the column of item ids
        interaction_col_names (list): specify the cols that represent user-item interactions 
        n_groups (int, optional): item groups per user. Defaults to 1.
        theta_max (int, optional): flipping threshold. Defaults to 1.
    """
    response_col_names = [col+'_r' for col in interaction_col_names]
    df = df.copy()
    n_items = df[item_col].nunique()
    n_features = len(response_col_names)
    df[response_col_names] = df[interaction_col_names].copy()
    user_groups = df.groupby(id_col)
    # for each user, assign each of their items to a group (max n_groups)
    df['group'] = user_groups[id_col].transform(lambda x: np.random.choice(n_groups, len(x)))
    # choose a beta max based on its user-wise distribution - the max fraction of rated movies
    # this ensures that we will not fill in more unrated items than the maximum rated by any user
    # this is just an assumption made; others can be made as well
    beta_max = user_groups.size().max()/n_items

    pars_cols = ['group', 'r']
    if is_fixed_response:
        # Algorithm 7
        # generate a beta for all users - the proportion of unrated items to be filled - not more than beta_max
        beta = np.random.uniform(0, beta_max)
#         df['r'] = df.groupby([id_col, 'group'])['group'].transform(lambda x: np.random.binomial(1, theta_max/2))
        df['r'] = df.groupby([id_col, 'group'])['group'].transform(lambda x: np.random.uniform())
        df['r'] = df['r'].apply(lambda x: 0 if x < theta_max else 1) 
        # PREVIOUSLY THERE WAS AN ERROR AND IT WAS 1 if else 0; it should be 0 if else 1
    else:
        # Algorithm 8
        # generate a beta for each user
        df['beta'] = user_groups[id_col].transform(lambda x: np.random.uniform(0, beta_max))
        pars_cols.append('beta', 'theta')

    # initialise user-wise data frame for each unrated movie: conserve the beta, sigma, treatment
    df_users_unrated = user_groups[pars_cols].agg('min')
    df_users_unrated['n_rated'] = user_groups.size()
    df_users_unrated['n_rated'].fillna(0)
    df_users_unrated.reset_index(inplace=True)
    # for each user, radomly choose a random number of unrated movie ids
    all_items = set(df[item_col].unique())
    df_users_unrated['unselected_set'] = user_groups[item_col].agg(list).apply(
        lambda s: list(all_items.difference(set(s)))).reset_index()[item_col]
    if is_fixed_response:
        # Algorithm 7
#         df_users_unrated['unselected_subset'] = df_users_unrated.apply(
#             lambda x: np.random.choice(x['unselected_set'], min(len(x['unselected_set']), int(beta*x['n_rated'])), replace=False), axis=1)
        df_users_unrated['unselected_subset'] = df_users_unrated.apply(
            lambda x: choose_from_unselected(x['unselected_set'], beta, x['n_rated']), axis=1)
    else:
        # Algorithm 8
#         df_users_unrated['unselected_subset'] = df_users_unrated.apply(
#             lambda x: np.random.choice(x['unselected_set'], min(len(x['unselected_set']), int(x['beta']*x['n_rated'])), replace=False), axis=1)
        df_users_unrated['unselected_subset'] = df_users_unrated.apply(
            lambda x: choose_from_unselected(x['unselected_set'], x['beta'], x['n_rated']), axis=1)
    df_users_unrated.drop(columns=['unselected_set'], inplace=True)
    df_users_unrated.rename(columns={'unselected_subset':item_col}, inplace=True)
    # explode the table for each user_id and unrated movie_id
    df_users_unrated = df_users_unrated.explode(item_col, ignore_index=True)

    # simulate ratings for unselected - generate random binary values for the unrated items
    df_users_unrated[interaction_col_names] = np.random.binomial(1, 0.5, (len(df_users_unrated), n_features))
    df_users_unrated[response_col_names] = df[interaction_col_names].copy()

    if is_fixed_response:
        # Algorithm 7
        df.loc[df['r'] == 1, response_col_names] = 1 - df.loc[df['r'] == 1, response_col_names]
        df_users_unrated.loc[df_users_unrated['r'] == 1, response_col_names] = 1 - df_users_unrated.loc[df_users_unrated['r'] == 1, response_col_names]
    else:
        # Algorithm 8
        # apply obfuscation to the original ratings
        df.loc[df['r'] >= df['theta'], response_col_names] = 1 - df.loc[df['r'] >= df['theta'], response_col_names]
        # apply obfuscation to the fictive ratings
        df_users_unrated.loc[df_users_unrated['r'] >= df_users_unrated['theta'], response_col_names] = 1 - df_users_unrated.loc[df_users_unrated['r'] >= df_users_unrated['theta'], response_col_names]

    # add the original/synthetic flag
    df['is_original'] = True
    df_users_unrated['is_original'] = False

    # concatenate the original and synthetic data with noise
    return df, df_users_unrated


def perturbation_no_masking(df:pd.DataFrame, is_fixed:bool, id_col:str, target_col:str, sigma_max:float):
    """
    Algorithms for perturbation of user-item ratings, without masking. 
    These correspond to Algorithms 1 and 2 in the original framework by Polat and Batmaz. 
    If fixed response, the same flipping threshold theta will be applied to the entire data frame. 
    If variable response, a different flipping threshold theta (not larger than theta_max) 
    is generated for each group of items, under each user.


    Args:
        df (pd.DataFrame): the data frame to be perturbed
        is_fixed (bool): perturbation type - fixed or variable
        id_col (str): specify the ID column in the data frame
        target_col (str): specify the ratings column, the one that will be targeted by the perturbation
        sigma_max (float): the variance of the noise distribution

    Returns:
        pd.DataFrame: perturbated data frame
    """
    df = df.copy()
    alpha = None
    if is_fixed:
        alpha = np.sqrt(3) * sigma_max
        df['z_{}_noise'.format(target_col)] = df['z_{}'.format(target_col)] + np.random.uniform(-alpha, alpha, df.shape[0])
    else:
        id_groups = df.groupby(id_col)
        # randomly choose a randomisation treatment for each user: coin flip between Gaussian and uniform
        df['treatment'] = id_groups[id_col].transform(lambda x: np.random.binomial(1, 0.5))
        # randomly choose a sigma for each user
        df['sigma'] = id_groups[id_col].transform(lambda x: np.random.uniform(0, sigma_max))
        # generate noise for the samples that receive the uniform random treatment
        df.loc[df['treatment'] == 0, 'noise'] = np.random.uniform(
            -np.sqrt(3)*df[df['treatment'] == 0]['sigma'], np.sqrt(3)*df[df['treatment'] == 0]['sigma'])
        # generate noise for the samples that receive the Gaussian random treatment
        df.loc[df['treatment'] == 1, 'noise'] = np.random.normal(0, df[df['treatment'] == 1]['sigma'])
        # add the noise
        df['z_rating_noise'] = df['z_rating'] + df['noise']
    df = transform_reverse_standardisation(df, target_col)
    return df


def perturbation_with_masking(df:pd.DataFrame, is_fixed:bool, id_col:str, target_col:str, item_col:str, sigma_max:float):
    """
    Algorithms for perturbation of user-item ratings, with masking. 
    This allows for concealling real user-item ratings, by generating user-item pairs and ratings based on
    those items that a user has not rated yet.
    These correspond to Algorithms 3 and 4 in the original framework by Polat and Batmaz. 
    If fixed response, the same flipping threshold theta will be applied to the entire data frame. 
    If variable response, a different flipping threshold theta (not larger than theta_max) 
    is generated for each group of items, under each user.


    Args:
        df (pd.DataFrame): the data frame to be perturbed
        is_fixed (bool): perturbation type - fixed or variable
        id_col (str): specify the ID column in the data frame
        target_col (str): specify the ratings column, the one that will be targeted by the perturbation
        item_col (str): specify the column of item ids
        sigma_max (float): the variance of the noise distribution
    """
    df = df.copy()
    n_items = df[item_col].nunique()
    
    # add the noise to the rated items
    df = perturbation_no_masking(df, is_fixed, id_col, target_col, sigma_max)
    
    id_groups = df.groupby(id_col)
    beta_max = id_groups.size().max()/n_items
    alpha = None
    pars_cols = [] if is_fixed else ['beta', 'sigma', 'treatment']
    if is_fixed:
        alpha = np.sqrt(3) * sigma_max
        beta = np.random.uniform(0, beta_max)
    else:
        df['beta'] = id_groups[id_col].transform(lambda x: np.random.uniform(0, beta_max))

    # initialise user-wise data frame for each unrated movie: conserve the beta, sigma, treatment
    df_users_unrated = pd.DataFrame() if is_fixed else id_groups[pars_cols].agg('min')
    df_users_unrated['n_rated'] = id_groups.size()
    df_users_unrated['n_rated'].fillna(0)
    # for each user, radomly choose a random number of unrated movie ids
    all_items = set(df[item_col].unique())
    df_users_unrated['unselected_set'] = id_groups[item_col].agg(list).apply(
        lambda s: list(all_items.difference(set(s)))).reset_index()[item_col]
    df_users_unrated.reset_index(inplace=True)

    # user-wise set of unselected movie-ids
    # pick a random subset of these unselected movies
    if is_fixed:
#         df_users_unrated = id_groups[item_col].agg(list).apply(lambda s: list(all_items.difference(set(s)))).reset_index()
#         df_users_unrated.columns = [id_col, 'unselected_set']
#         df_users_unrated['unselected_subset'] = df_users_unrated.apply(
#             lambda x: np.random.choice(x['unselected_set'], int(beta*len(x['unselected_set'])), replace=False), axis=1)
        df_users_unrated['unselected_subset'] = df_users_unrated.apply(
            lambda x: choose_from_unselected(x['unselected_set'], beta, x['n_rated']), axis=1)
    else:
#         df_users_unrated = id_groups[['beta', 'sigma', 'treatment']].agg('min').reset_index()
#         df_users_unrated['unselected_set'] = id_groups[item_col].agg(list).apply(
#             lambda s: list(all_items.difference(set(s)))).reset_index()[item_col]
#         df_users_unrated['unselected_subset'] = df_users_unrated.apply(
#             lambda x: np.random.choice(x['unselected_set'], int(x['beta']*len(x['unselected_set'])), replace=False), axis=1)
        df_users_unrated['unselected_subset'] = df_users_unrated.apply(
            lambda x: choose_from_unselected(x['unselected_set'], x['beta'], x['n_rated']), axis=1)

    df_users_unrated.drop(columns=['unselected_set'], inplace=True)
    df_users_unrated.rename(columns={'unselected_subset':item_col}, inplace=True)
    # explode the table for each user_id and unrated movie_id => a data frame of users and a random selection of their unrated movies
    df_users_unrated = df_users_unrated.explode(item_col, ignore_index=True)

    # Generate random ratings for the unrated items
    ratings_distribution = list((df[target_col].value_counts()/len(df)).sort_index())
    df_users_unrated[target_col] = np.random.choice(list(range(1,6)), len(df_users_unrated), ratings_distribution)
    # normalise ratings
    df_users_unrated = transform_standardize(df_users_unrated, id_col, target_col)
    # user-wise ratings descriptive stats (mean, std)
    df_user_ratings_desc_stats = df.groupby(id_col)[['user_{}_mean'.format(target_col), 'user_{}_std'.format(target_col)]].agg(min).reset_index()
    df_users_unrated = df_users_unrated.merge(df_user_ratings_desc_stats, on=id_col, how='left')
    # add the noise to the synthetic items
    df_users_unrated = perturbation_no_masking(df_users_unrated, is_fixed, id_col, target_col, sigma_max)

    # TODO if we use a per-user descriptive stats df, then the transform reverse standardisation util function will do the merge
    # hence everything below will be more contrived
    # de-standardise
#     df = transform_reverse_standardisation(df, target_col)
    # De-normalise syntehtially rated items with noise - we need the mean and std from the rated items
#     df_users_unrated = transform_reverse_standardisation(df_users_unrated, target_col)

    # add the original/synthetic flag
    df['is_original'] = True
    df_users_unrated['is_original'] = False
    
    return df, df_users_unrated