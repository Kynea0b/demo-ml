# 概要

回帰分析が使用できる例を挙げています。

「1990年の米国国勢調査から得られたカリフォルニア州の住宅価格」の表形式データセット（＝構造化データセット）を使用して回帰分析しています。

このデータセットで使用されるデータは、国勢調査から得られた各世帯ごとの、年収中央値、築年数、部屋数、寝室数、などで構成されており、住宅価格を目的変数としたとき、
- 目的変数が正規分布を仮定できそう
- 各説明変数は独立を仮定して良さそう

回帰分析の対象として妥当なデータであると言うことが言えそうです。
このためこのデータを用いて演習に回帰分析してみます。

## 単回帰分析と重回帰分析

カリフォルニアデータセットを例にして、単回帰分析と重回帰分析を行います。
説明変数のピックアップから説明変数と目的変数の関係の分析について、その技法をまとめます。

ここでは単回帰分析と重回帰分析について扱います。

pythonのカリフォルニア住宅価格を例に、回帰分析の技法をまとめます。
ここでは目的変数を`HousingPrices`として、説明変数をどう選ぶから見ていきます。
今回、pythonによるデータの取得およびデータ間の関係に潜む特徴を可視化には`seaborn`、`sklearn`を使用しています。

- [searborn.heatmap](https://seaborn.pydata.org/generated/seaborn.heatmap.html)
- [sklearn.datasets.fetch_california_housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)

## 単回帰分析

source-code: [simple-regression.py](https://github.com/kokeshiM0chi/demo-datascience/blob/main/sample-linear-regression/simple-regression.py)

featuresで説明変数を全て列挙して、各相関係数を計算し、ヒートマップで出力します。

<img width="400" alt="heatmap" src="https://github.com/user-attachments/assets/e0f7d023-8d7d-42c6-a629-00db23e663fd">

赤枠内を見ると、目的変数の`HousingPrices`と相関係数が高いのは`MedInc`です。

この章では目的変数を`HousingPrices`として、`MedInc`を説明変数として単回帰分析を行います。

説明変数と目的変数の散布図を取得します。

<img width="400" alt="scatter" src="https://github.com/user-attachments/assets/6790363a-33bf-4900-a408-1146e672c465">

`LinearRegression()`に説明変数と目的変数を割り当てて、線形回帰モデルを取得します。

<img width="400" alt="scatter-regression" src="https://github.com/user-attachments/assets/70d81a5f-2e12-45e0-a6c3-fb4706cb2983">

取得したモデルの精度を確認します。以下のような結果を得ました。

回帰直線の切片 1.0718214790675578
回帰係数 0.327793797528796
決定係数 0.36060120745920043
回帰直線 y =  0.327793797528796 x +  1.0718214790675578

## 重回帰分析

source code: https://github.com/kokeshiM0chi/demo-datascience/blob/main/sample-linear-regression/multiple-regression.py

続いて重回帰分析です。相関係数の低いものも含めて、ヒストグラムにあるもの全てを説明変数とします。
Lasso回帰を行い、その後Ridge回帰をしています。

結果性能は以下のようになりました。

### Lasso回帰

正則化項のalphaの値を0.3以上にしたところ、回帰係数が全て0になったため、0.3で行いました。`MedInc`以外は０となり、`MedInc`が説明変数として影響が大きいようです。
決定係数が、あまり大きくなりませんが、これは、

```
Lasso; alpha =  0.3
MedInc :  0.397
HouseAge:  0.000
AveRooms:  0.000
AveBedrms:  0.000
Population: -0.000
AveOccup: -0.000
Latitude: -0.000
Longitude: -0.000
L1ノルム 0.39674612499158035
訓練データに対するMSE: 0.731
テストデータに対するMSE: 0.710
決定係数: 0.352
```

### Ridge回帰

考慮すべき説明変数が多くないので、Ridge回帰の方が適合度が高そうです。決定係数も0.645の結果です。
平均二乗誤差も訓練データ、テストデータともにラッソ回帰よりもよい値です。

```
Ridge; alpha =  1.0
MedInc :  0.777
HouseAge:  0.123
AveRooms: -0.172
AveBedrms:  0.123
Population:  0.034
AveOccup: -0.264
Latitude: -0.846
Longitude: -0.768
L2ノルム: 1.428
訓練データに対するMSE: 0.400
テストデータに対するMSE: 0.391
決定係数:0.645
```





