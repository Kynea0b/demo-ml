import pandas as pd

# from sklearn.datasets import load_boston
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# カリフォルニアデータセットの読み込み
california_housing = fetch_california_housing()

# データフレームの作成
df = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
# 目的変数(PRICE)をデータフレームに結合
df["PRICE"] = np.array(california_housing.target)
# データセットの変数の説明の表示
print(california_housing["DESCR"])

# pandasのscatter_matrix関数を使いペアプロットを表示
from pandas.plotting import scatter_matrix

fig = pd.plotting.scatter_matrix(df, figsize=(15, 15))

# plt.show()

###########################################
# モデルを作成して、目的変数と説明変数の回帰係数を知る
###########################################

from sklearn.model_selection import train_test_split

# 説明変数の抽出(CRIME、RM)
data = df.loc[:, ["AveRooms", "HouseAge"]].values
# 目的変数の抽出
target = df.loc[:, "PRICE"].values
# データの分割(訓練データ、検証データ)
X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=0)

# sklearnモジュールからLinearRegressionをインポート
from sklearn.linear_model import LinearRegression

# 線形モデルのオブジェクト生成
model = LinearRegression()
# モデルの学習
model.fit(X_train, y_train)

###########################################
# - モデルを使用した、犯罪率と住宅価格の関係
#
###########################################
print("")
print(model.predict([[4, 40]]))
print(model.predict([[4, 5]]))
