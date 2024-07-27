# 単回帰分析と重回帰分析

カリフォルニアデータセットを例にして、単回帰分析と重回帰分析を行います。
説明変数のピックアップから説明変数と目的変数の関係の分析について、その技法をまとめます。
また、その技法の背景にある数学理論などにも触れながら、その技法によって導かれた分析結果についてどこまで妥当と言えるのかについて理論的な評価を行いながら分析を進めます。

ここでは単回帰分析と重回帰分析について扱います。

pythonのカリフォルニア住宅価格を例に、回帰分析の技法をまとめます。
ここでは目的変数を`HousingPrices`として、説明変数をどう選ぶから見ていきます。
今回、pythonによるデータの取得およびデータ間の関係に潜む特徴を可視化には`seaborn`、`sklearn`を使用しています。

- [searborn.heatmap](https://seaborn.pydata.org/generated/seaborn.heatmap.html)
- [sklearn.datasets.fetch_california_housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)

## 単回帰分析

[simple-regression.py](https://github.com/kokeshiM0chi/demo-datascience/blob/main/sample-linear-regression/simple-regression.py)
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

続いて重回帰分析です。相関係数の低いものも含めて、ヒストグラムにあるもの全てを説明変数とします。



