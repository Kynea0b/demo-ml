import math
import numpy as np
import pandas as pd
import scipy.stats

# dataset: https://www.kaggle.com/datasets/samtyagi/audacity-ab-testing

df=pd.read_csv("./data/homepage_actions.csv")
print(df.head())

group=df.groupby('group').count()
print(group)

data=np.matrix([[932,3332],[928,2996]])
chi2,p,ddof,expected=scipy.stats.chi2_contingency(data,correction=False)

print("カイ二乗値:", chi2)
print("p値:", p)
print("自由度:", ddof)
print("期待度数:", expected)