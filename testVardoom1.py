import preprocessing
import seaborn as sns
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Retrieve train & test from preprocessing.py
train = preprocessing.train
test = preprocessing.test
data = preprocessing.data
meta = preprocessing.meta
train_size = preprocessing.train_size
target = preprocessing.target

# Getting dummies for all possible remaining categories
dummiesList = ['ps_ind_01', 'ps_ind_03', 'ps_ind_14', 'ps_ind_15', 'ps_car_11', 'ps_calc_04', 'ps_calc_05',
               'ps_calc_06', 'ps_calc_07', 'ps_calc_08', 'ps_calc_09', 'ps_calc_10', 'ps_calc_11', 'ps_calc_12',
               'ps_calc_13', 'ps_calc_14']
data = pd.get_dummies(data, columns=dummiesList, drop_first=True)

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


# ================== Step 5: Machine Learning ==================
def adaboost(x_train, y_train, kfold=StratifiedKFold(n_splits=10), n_jobs=1, ver=0):
    """ Adaboost algorithm """
    algoInt = DecisionTreeClassifier()
    algo = AdaBoostClassifier(base_estimator=algoInt, random_state=7)
    param_grid = {"algorithm": ["SAMME.R", "SAMME"],
                  "base_estimator__criterion": ["entropy", "gini"],
                  "base_estimator__splitter": ["best"],
                  'base_estimator__max_depth': [2, 6, 10],
                  "learning_rate": [1.3, 1.5, 1.7],
                  "n_estimators": [4, 5, 6]}
    result = GridSearchCV(algo, param_grid=param_grid, cv=kfold, scoring="accuracy", n_jobs=n_jobs, verbose=ver)
    result.fit(x_train, y_train)
    return result


res = adaboost(x_train=train, y_train=target, n_jobs=-1, ver=0)
targetTest = res.predict(test)
