# ====================== Step 0: Import library ======================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, GridSearchCV

seed = 45

# ====================== Step 1: Read Data Set ======================
print("Read Data Set")
train = pd.read_csv('input/train.csv', na_values=-1)
test = pd.read_csv('input/test.csv', na_values=-1)

plt.figure(figsize=(10, 3))
sns.countplot(train['target'])
plt.xlabel('Target')
plt.show()

cor = train.corr()
plt.figure(figsize=(16, 10))
sns.heatmap(cor, cmap='Set3')

# ====================== Step 2: Clean Data Set ======================
# ps_calc_* has no relation with other variables
print("Drop ps_calc_*")
ps_calc = train.columns[train.columns.str.startswith('ps_calc')]
train = train.drop(ps_calc, axis=1)
test = test.drop(ps_calc, axis=1)

# Missing value in Data Set
print("Missing Value")


def deleteMissingValue(data):
    columns = data.columns
    for col in columns:
        if data[col].isnull().sum() > 0:
            data[col].fillna(data[col].mode()[0], inplace=True)


deleteMissingValue(train)
deleteMissingValue(test)

# Convert variables into category type
print("Category Type")


def changeIntoCategoryType(data):
    columns = data.columns
    for col in columns:
        if data[col].nunique() <= 104:
            data[col] = data[col].astype('category')


changeIntoCategoryType(train)
changeIntoCategoryType(test)

# Variables Analysis
print("Variables analysis")
cat_col = [col for col in train.columns if '_cat' in col]
bin_col = [col for col in train.columns if 'bin' in col]
tot_cat_col = list(train.select_dtypes(include=['category']).columns)
other_cat_col = [c for c in tot_cat_col if c not in cat_col + bin_col]
other_cat_col
num_col = [c for c in train.columns if c not in tot_cat_col]
num_col.remove('id')
num_col

# ====================== Step 3: Determine outliers in the Data Set ======================
print("Outliers")


def outlier(data, columns):
    for col in columns:
        quartile_1, quartile_3 = np.percentile(data[col], [25, 75])
        quartile_f, quartile_l = np.percentile(data[col], [1, 99])
        IQR = quartile_3 - quartile_1
        lower_bound = quartile_1 - (1.5 * IQR)
        upper_bound = quartile_3 + (1.5 * IQR)
        print(col, lower_bound, upper_bound, quartile_f, quartile_l)

        data[col].loc[data[col] < lower_bound] = quartile_f
        data[col].loc[data[col] > upper_bound] = quartile_l


outlier(train, num_col)
outlier(test, num_col)

# ====================== Step 4: One Hot Encoding ======================
print("One Hot Encoding")


def OHE(df1, df2, column):
    cat_col = column
    len_df1 = df1.shape[0]
    df = pd.concat([df1, df2], ignore_index=True)
    c2, c3 = [], {}
    print('Categorical feature', len(column))
    for c in cat_col:
        if df[c].nunique() > 2:
            c2.append(c)
            c3[c] = 'ohe_' + c
    df = pd.get_dummies(df, prefix=c3, columns=c2, drop_first=True)
    df1 = df.loc[:len_df1 - 1]
    df2 = df.loc[len_df1:]
    print('Train', df1.shape)
    print('Test', df2.shape)
    return df1, df2


train1, test1 = OHE(train, test, tot_cat_col)

# ====================== Step 5: Split Data Set ======================
print("Split Data")
X = train1.drop(['target', 'id'], axis=1)
y = train1['target'].astype('category')
x_test = test1.drop(['target', 'id'], axis=1)
del train1, test1

# ====================== Step 6: Hyperparameter tuning ======================
print("Hyperparameter Tuning")
# Grid search
logreg = LogisticRegression(class_weight='balanced')
param = {'C': [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1]}
clf = GridSearchCV(logreg, param, scoring='roc_auc', refit=True, cv=3, n_jobs=8)
clf.fit(X, y)
print('Best roc_auc: {:.4}, with best C: {}'.format(clf.best_score_, clf.best_params_['C']))

# ====================== Step 7: Logistic Regression model ======================
print("Logic Regression")
kf = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)
pred_test_full = 0
cv_score = []
i = 1
for train_index, test_index in kf.split(X, y):
    print('\n{} of kfold {}'.format(i, kf.n_splits))
    xtr, xvl = X.loc[train_index], X.loc[test_index]
    ytr, yvl = y[train_index], y[test_index]

    lr = LogisticRegression(class_weight='balanced', C=0.005)
    lr.fit(xtr, ytr)
    pred_test = lr.predict_proba(xvl)[:, 1]
    score = roc_auc_score(yvl, pred_test)
    print('roc_auc_score', score)
    cv_score.append(score)
    pred_test_full += lr.predict_proba(x_test)[:, 1]
    i += 1

proba = lr.predict_proba(xvl)[:, 1]
fpr, tpr, threshold = roc_curve(yvl, proba)
auc_val = auc(fpr, tpr)

plt.figure(figsize=(14, 8))
plt.title('Reciever Operating Charactaristics')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % auc_val)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')

# ====================== Step 8: Predict for unseen Data Set ======================
print("Prediction")
y_pred = pred_test_full / 5
submit = pd.DataFrame({'id': test['id'], 'target': y_pred})
submit.to_csv('output/simpleLogisticModel.csv', index=False)
