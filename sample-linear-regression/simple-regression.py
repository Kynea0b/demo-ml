import pandas as pd
from IPython.display import display
import seaborn as sns
# from sklearn.datasets import load_boston
import numpy as np
import matplotlib.pyplot as plt
# データをインポートしてDataFrameの内容とデータ型、欠損の確認をする
from sklearn.datasets import fetch_california_housing
california_housing_data = fetch_california_housing()
# 説明変数候補
exp_df = pd.DataFrame(california_housing_data.data, columns=california_housing_data.feature_names)
exp_df.to_csv('data/california.csv')
# 目的変数
target_df = pd.DataFrame(california_housing_data.target, columns=['HousingPrices'])
# データを結合
data = pd.concat([exp_df, target_df], axis=1)
display(data.head())
print(data.shape) # (20640, 9)
print(data.dtypes) # 全てfloat64
print(data.isnull().sum()) # 全て0


# *** パラメータを満遍なく観察 ***
# ヒートマップを表示
plt.figure(figsize=(12, 9))
sns.heatmap(data.corr(), annot=True, cmap='Greens', fmt='.2f', linewidths=.5)
plt.savefig('california_housing_heatmap.png')


# *** 目的変数と説明変数の抽出 ****
exp_var = 'MedInc'
tar_var = 'HousingPrices'

# *** 目的変数と説明変数の関係を観察 ***
# 散布図を表示
plt.figure(figsize=(12, 9))
plt.scatter(data[exp_var], data[tar_var])
plt.xlabel(exp_var)
plt.ylabel(tar_var)
plt.title('california_housing_scatter')
plt.savefig('california_housing_scatter.png')
data[[exp_var, tar_var]].describe()

# 外れ値を除去
q_98 = data[exp_var].quantile(0.98)
q_02 = data[exp_var].quantile(0.02)
print('98%点の分位数', q_98)

# 絞り込む; 目的変数の値
data = data[data[tar_var] < q_98]
data = data[data[tar_var] > q_02]

# 散布図を表示
plt.figure(figsize=(12, 9))
plt.scatter(data[exp_var], data[tar_var])
plt.xlabel(exp_var)
plt.ylabel(tar_var)
plt.title('california_housing_scatter2')
plt.savefig('california_housing_scatter2.png')
plt.show()

# 記述統計量を確認
data[[exp_var, tar_var]].describe()

# 説明変数と目的変数にデータを分割
X = data[[exp_var]]
print(data.shape)
#display(X.head())
print(X.head())
y = data[[tar_var]]
print(y.shape)
#display(y.head())
print(y.head())

# 単回帰モデルを学習する
# 学習
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)

# 精度の確認
print('回帰直線の切片', model.intercept_[0])
print('回帰係数', model.coef_[0][0])
print('決定係数', model.score(X, y))
print('回帰直線', 'y = ', model.coef_[0][0], 'x + ', model.intercept_[0])

# 回帰直線と散布図を表示
plt.figure(figsize=(12, 9))
plt.scatter(X, y)
plt.plot(X, model.predict(X), color='red')
plt.xlabel(exp_var)
plt.ylabel(tar_var)
plt.title('california_housing_regression')
plt.savefig('california_housing_regression.png')
