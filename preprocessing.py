import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import VarianceThreshold

"""This file preprocesses the data. It is based on the kernel of Bert Carremans, Data Preparation & Exploration"""

# ================== Step 0: Import Data and Merge ==================

# Loading data
train = pd.read_csv("input/train.csv")
test = pd.read_csv("input/test.csv")

# Merging train and test
train_size = train.shape[0]
test_size = test.shape[0]
target = train["target"]
train.drop("target", inplace=True, axis=1)
data = pd.concat([train, test])
data_size = data.shape[0]

# ================== Step 1: Metadata ==================
metadata = []
for f in data.columns:
    # Defining the role
    if f == 'target':
        role = 'target'
    elif f == 'id':
        role = 'id'
    else:
        role = 'input'

    # Defining the level
    if 'bin' in f or f == 'target':
        level = 'binary'
    elif 'cat' in f or f == 'id':
        level = 'nominal'
    elif data[f].dtype == float:
        level = 'interval'
    elif data[f].dtype == int:
        level = 'ordinal'

    # Initialize keep to True for all variables except for id
    keep = True
    if f == 'id':
        keep = False

    # Defining the data type
    dtype = data[f].dtype

    # Creating a Dict that contains all the metadata for the variable
    f_dict = {'varname': f, 'role': role, 'level': level, 'keep': keep, 'dtype': dtype}
    metadata.append(f_dict)

meta = pd.DataFrame(metadata, columns=['varname', 'role', 'level', 'keep', 'dtype'])
meta.set_index('varname', inplace=True)

# ================== Step 2: Data Quality Checks ==================

# Dropping the variables with too many missing values (> 40% missing values)
vars_to_drop = ['ps_car_03_cat', 'ps_car_05_cat']
data.drop(vars_to_drop, inplace=True, axis=1)
meta.loc[vars_to_drop, 'keep'] = False  # Updating the meta

# Replacing -1 with the mean or the mode
mean_imp = Imputer(missing_values=-1, strategy='mean', axis=0)
mode_imp = Imputer(missing_values=-1, strategy='most_frequent', axis=0)
data['ps_reg_03'] = mean_imp.fit_transform(data[['ps_reg_03']]).ravel()
data['ps_car_12'] = mean_imp.fit_transform(data[['ps_car_12']]).ravel()
data['ps_car_14'] = mean_imp.fit_transform(data[['ps_car_14']]).ravel()
data['ps_car_11'] = mode_imp.fit_transform(data[['ps_car_11']]).ravel()


# Dealing with ps_car_11_cat which has 104 distinct values
def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


def target_encode(trn_series=None, tst_series=None, target=None, min_samples_leaf=1, smoothing=1, noise_level=0):
    """
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior
    """
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)


train_encoded, test_encoded = target_encode(trn_series=train["ps_car_11_cat"], tst_series=test["ps_car_11_cat"],
                                            target=target, min_samples_leaf=100, smoothing=10, noise_level=0.01)
data['ps_car_11_cat_te'] = pd.concat([train_encoded, test_encoded])
data.drop('ps_car_11_cat', axis=1, inplace=True)
meta.loc['ps_car_11_cat', 'keep'] = False  # Updating the meta

# ================== Step 3: Feature Engineering ==================

# Getting dummies for other cat columns
data = pd.get_dummies(data, columns=meta[(meta.level == 'nominal') & meta.keep].index, drop_first=True)


# ================== Step 4: Output Data ==================
train = data.iloc[:train_size, :]
train['target'] = target
test = data.iloc[train_size:, :]
