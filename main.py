import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('input/train.csv', na_values=-1)
test = pd.read_csv('input/test.csv', na_values=-1)

# 0/1
plt.figure(figsize=(10, 3))
sns.countplot(train['target'])
plt.xlabel('Target')
plt.show()

# Correlation
cor = train.corr()
plt.figure(figsize=(16, 10))
sns.heatmap(cor, cmap='Set3')

