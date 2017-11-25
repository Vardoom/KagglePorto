import preprocessing
import seaborn as sns
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

# Retrieve train & test from preprocessing.py
train = preprocessing.train
# test = preprocessing.test
# data = preprocessing.data
# meta = preprocessing.meta
train_size = preprocessing.train_size
target = preprocessing.target


# ================== Step 5: Machine Learning ==================

x_train = train.iloc[:train_size // 2, :]
y_train = target.iloc[:train_size // 2]
x_test = train.iloc[train_size // 2:, :]
y_test_true = target.iloc[train_size // 2:]
algorithm = ExtraTreesClassifier(n_jobs=-1)
scores = cross_val_score(algorithm, train, target, cv=10, scoring='roc_auc')
