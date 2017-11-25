import preprocessing
import seaborn as sns
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

# Retrieve train & test from preprocessing.py
train = preprocessing.train.drop(labels='target', axis=1, inplace=False)
train_size = preprocessing.train_size
target = preprocessing.target
# test = preprocessing.test
# data = preprocessing.data
# meta = preprocessing.meta


# ================== Step 5: Machine Learning ==================
def method_1(estimator=ExtraTreesClassifier()):
    scores = cross_val_score(estimator=estimator, X=train, y=target, cv=5, scoring='roc_auc', n_jobs=-1)
    return scores