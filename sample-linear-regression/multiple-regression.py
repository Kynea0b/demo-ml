# https://zenn.dev/omochimaru/articles/9b289f4a9455b7
import pandas as pd
from IPython.display import display
import seaborn as sns

# from sklearn.datasets import load_boston
import numpy as np
import matplotlib.pyplot as plt

# データをインポートしてDataFrameの内容とデータ型、欠損の確認をする
from sklearn.datasets import fetch_california_housing

california_housing_data = fetch_california_housing()

# *** データの取得 ***
# 説明変数候補
exp_df = pd.DataFrame(
    california_housing_data.data, columns=california_housing_data.feature_names
)
exp_df.to_csv("data/california.csv")
# 目的変数
target_df = pd.DataFrame(california_housing_data.target, columns=["HousingPrices"])
# データを結合
data = pd.concat([exp_df, target_df], axis=1)
display(data.head())
print(data.shape)  # (20640, 9)
print(data.dtypes)  # 全てfloat64
print(data.isnull().sum())  # 全て0


# 説明変数
exp_vars = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]
# 目的変数
tar_var = "HousingPrices"
# 外れ値を除去
for exp_var in exp_vars:
    q_98 = data[exp_var].quantile(0.98)
    print(exp_var, "98%点の分位数", q_98)
    data = data[data[exp_var] < q_98]
    # display(data[[exp_var, tar_var]].describe())
# 説明変数と目的変数にデータを分割
X = data[exp_vars]
print(data.shape)
display(X.head())
y = data[[tar_var]]
print(y.shape)
display(y.head())
# 訓練データとテストデータに分割
# 分割のためのモジュールをインポート
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print("X_train", X_train.shape)
print("y_train", y_train.shape)
print("X_test", len(X_test))
print("y_test", len(y_test))
#  X_trainを標準化する
# 標準化ライブラリをインポート
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
print("X_train_scaled", X_train_scaled.shape)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=exp_vars)
print("----StandardScaler----")
display(X_train_scaled.head())
# 学習
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train_scaled, y_train)
# 予測値を計算
y_pred = model.predict(X_train_scaled)
print("--予測値を計算--")
print(y_pred[:10])
# 標準偏回帰係数を計算
print("--標準偏回帰係数を計算--model.coef_")

# ---- WIP 偏回帰係数グラフ化 TODO ----
# 重み（偏回帰係数）を計算
model.coef_
# 重みをデータフレームとして表示
coef = model.coef_
df_coef = pd.DataFrame(coef).T


# df_coef.columns = exp_df.columns[0:]


# ---- 偏回帰係数グラフか end ----
print(model.coef_)
print("--標準偏回帰係数を計算--")
for xi, wi in zip(exp_vars, model.coef_[0]):
    print("{0:7s}: {1:6.3f}".format(xi, wi))
# 係数を降順に並び替える（係数が大きいと回帰直線に大きな影響を与えていると言える）
import numpy as np

n_list = []
print("--標準偏回帰係数(係数を降順に並び替え)--")
for i in np.argsort(-model.coef_):
    n_list = list(i)
    for j in n_list:
        print("{0:7s}: {1:6.3f}".format(exp_vars[j], model.coef_[0][j]))
# 精度の確認
# 決定係数
print("決定係数:{:.3f}".format(model.score(X_train_scaled, y_train)))
from sklearn.metrics import mean_squared_error

# 訓練データに対する Mean Squared Error (MSE)
mse_train = mean_squared_error(y_train, y_pred)
print("訓練データに対するMSE: {:.3f}".format(mse_train))

# テストデータに対する MSE
X_test_scaled = scaler.transform(
    X_test
)  # テストデータを訓練データから得られた平均と標準偏差で標準化
y_test_pred = model.predict(X_test_scaled)  # テストデータに対して予測する
mse_test = mean_squared_error(y_test, y_test_pred)
print("テストデータに対するMSE: {:.3f}".format(mse_test))

# Lasso回帰
print("--------------Lasso回帰---------------")
from sklearn.linear_model import Lasso
alpha = 0.3
print("Lasso; alpha = ", alpha)
lasso = Lasso(alpha=alpha)
lasso.fit(X_train_scaled, y_train)
lasso_y_pred = lasso.predict(X_train_scaled)

# 偏回帰係数の確認
lasso_w = pd.Series(index=exp_vars, data=lasso.coef_)
for xi, wi in zip(exp_vars, lasso.coef_):  # coef_は回帰直線の傾き
    # print(f'{xi, wi}')
    print("{0:7s}: {1:6.3f}".format(xi, wi))
print("L1ノルム", np.linalg.norm(lasso_w))

lasso_mse_train = mean_squared_error(y_train, lasso_y_pred)
print("訓練データに対するMSE: {:.3f}".format(lasso_mse_train))

lasso_X_test_scaled = scaler.transform(X_test)
lasso_y_pred_test = lasso.predict(lasso_X_test_scaled)
lasso_mse_test = mean_squared_error(y_test, lasso_y_pred_test)
print("テストデータに対するMSE: {:.3f}".format(lasso_mse_test))

# 決定係数
print("決定係数: {:.3f}".format(lasso.score(lasso_X_test_scaled, y_test)))

# Ridge回帰
print("--------------Ridge回帰---------------")
from sklearn.linear_model import Ridge
alpha = 1.0
print("Ridge; alpha = ", alpha)
ridge = Ridge(alpha=alpha)
ridge.fit(X_train_scaled, y_train)
ridge_y_pred = ridge.predict(X_train_scaled)

# 偏回帰係数の確認
ridge_w = pd.DataFrame(ridge.coef_.T, index=exp_vars, columns=["Ridge"])
for xi, wi in zip(exp_vars, ridge.coef_[0]):
    print("{0:7s}: {1:6.3f}".format(xi, wi))
print("L2ノルム: {:.3f}".format(np.linalg.norm(ridge.coef_)))

# 訓練データに対する Mean Squared Error (MSE)
ridge_mse_train = mean_squared_error(y_train, ridge_y_pred)
print("訓練データに対するMSE: {:.3f}".format(ridge_mse_train))

# テストデータに対する MSE
ridge_y_test_pred = ridge.predict(X_test_scaled)  # テストデータに対して予測する
ridge_mse_test = mean_squared_error(y_test, ridge_y_test_pred)
print("テストデータに対するMSE: {:.3f}".format(ridge_mse_test))

# 決定係数
print("決定係数:{:.3f}".format(ridge.score(X_train_scaled, y_train)))


# Summary
print("-----compare Lasso and Ridge-----")
data = {
    "訓練データMSE": [mse_train, ridge_mse_train, lasso_mse_train],
    "テストデータMSE": [mse_test, ridge_mse_test, lasso_mse_test],
    "決定係数": [
        model.score(X_test_scaled, y_test),
        ridge.score(X_test_scaled, y_test),
        lasso.score(X_test_scaled, y_test),
    ],
}
df_mse = pd.DataFrame(data=data, index=["重回帰", "Ridge回帰", "Lasso回帰"])
print(df_mse)
