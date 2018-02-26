import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer

"""This file preprocesses the data."""

# ================== Step 0: Import Data and Merge ==================

# Loading data
train = pd.read_csv("input/train.csv")
test = pd.read_csv("input/test.csv")
print("Data loaded")

# Merging train and test
train_size = train.shape[0]
test_size = test.shape[0]
target = train["target"]
train.drop("target", inplace=True, axis=1)
data = pd.concat([train, test])
data_size = data.shape[0]
print("End of step 0 of preprocessing")

# ================== Step 1: Metadata ==================
metadata = []
metadataUpdated = []
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
    metadata.append({'varname': f, 'role': role, 'level': level, 'keep': keep, 'dtype': dtype})

meta = pd.DataFrame(metadata, columns=['varname', 'role', 'level', 'keep', 'dtype'])
meta.set_index('varname', inplace=True)
print("End of step 1 of preprocessing")

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
print("End of step 2 of preprocessing")

# ================== Step 3: Feature Engineering ==================

# Getting dummies for other cat columns
data = pd.get_dummies(data, columns=meta[(meta.level == 'nominal') & meta.keep].index, drop_first=True)
print("End of step 3 of preprocessing")


# Getting dummies for all possible remaining categories
dummiesList = ['ps_ind_01', 'ps_ind_03', 'ps_ind_14', 'ps_ind_15', 'ps_car_11', 'ps_calc_04', 'ps_calc_05',
               'ps_calc_06', 'ps_calc_07', 'ps_calc_08', 'ps_calc_09', 'ps_calc_10', 'ps_calc_11', 'ps_calc_12',
               'ps_calc_13', 'ps_calc_14']
data = pd.get_dummies(data, columns=dummiesList, drop_first=True)

""""
# Sorting continuous values into categories
catList = ['ps_reg_01', 'ps_reg_02', 'ps_reg_03', 'ps_car_12', 'ps_car_13', 'ps_car_14', 'ps_car_15', 'ps_calc_01',
           'ps_calc_02', 'ps_calc_03']
catListNew = [name + "_new" for name in catList]
for name in catList:
    column = pd.Series(data[name])
    percentage_rank = column.rank(method="max", pct=True)
    newName = name + "_new"
    data[newName] = 0
    data[newName][(percentage_rank > 0.25) & (percentage_rank <= 0.5)] = 1
    data[newName][(percentage_rank > 0.5) & (percentage_rank <= 0.75)] = 2
    data[newName][(percentage_rank > 0.75)] = 3
data = pd.get_dummies(data, columns=catListNew, drop_first=True)
data.drop(catList, inplace=True, axis=1)
meta.loc[catList, 'keep'] = False  # Updating the metadata
train = data.iloc[:train_size, :]
test = data.iloc[train_size:, :]
print("Data formated")
"""


# ================== Step 4: Output Data ==================
train = data.iloc[:train_size, :]
train = pd.concat([train, target], axis=1)
test = data.iloc[train_size:, :]
print("End of step 4 of preprocessing")

print("End of preprocessing")
