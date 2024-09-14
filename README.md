# demo-ml

このリポジトリは機械学習や統計分析のデモです。機械学習や統計学の手法を用いて、データの予測や分類を行います。また、kaggleコンテスト入賞時のデータ分析も掲載しています。
実装とともに、手法の妥当性などについて考察も書いています。

使用している手法リスト：
- 線形回帰
- 一般化線形回帰
- SVM、Randomforest
- Prophet
- Autoencoder
- Recommend


以下、各ディレクトリに格納されているプロジェクトの説明です。

## Kaggle Competition

[WiDS2021_Kaggle_result_TNN](https://github.com/kokeshiM0chi/demo-datascience/tree/main/WiDS2021_Kaggle_result_TNN)

Prophetを使用して、休日や季節変動のある和菓子の売り上げについて、売り上げ予測をしています。

## 線形回帰 (LM)

[sample-linear-regression](https://github.com/kokeshiM0chi/demo-datascience/tree/main/sample-linear-regression)

カリフォルニアデータセットをもとに線形回帰から、重回帰分析までを行います。

## 一般化線形回帰(GLM)からランダムフォレスト、SVNまで

[sample-improve-clicknum](https://github.com/kokeshiM0chi/demo-datascience/tree/main/sample-improve-clicknum)

一般化線形回帰から、ランダムフォレストやSVMまでを使用して、クリック数とデザイン要素の関係を調べます。

## Autoencoderによる手書き文字分類

[sample-autoencoder](https://github.com/kokeshiM0chi/demo-ml/tree/main/sample-autoencoder)

kerasを使用しています。


## [WIP] 食べログから抽出したテキストデータを形態素解析して類似度判定（Neologd使用）


## [WIP] Alibabaが採用するオンラインディスプレイ広告のCTR予測手法DINの実装

CTRを改善するためにアリババ社で使用されているDIN
[Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/abs/1706.06978)
- ユーザの特定の広告の履歴行動から、ユーザの関心を適応的に学習してベクトル表現を与えるローカルアクティベーションユニットを構築
- 上記のベクトル表現を広告ごとに異なるものにすることで、ユーザの興味関心を幅広く捉えて、モデルの表現能力を豊かにしている
- さらにこのDINにおいて使用される数百まんの学習パラメータを学習するための工夫も行っている
  
参考文献:
- https://arxiv.org/abs/1706.06978

## [WIP] Apacheでアクセスログの解析

[sample-apache-log](https://github.com/kokeshiM0chi/demo-datascience/tree/main/sample-apache-log)

作成中です。botと人間のアクセスログの比較を行い、自動分類するモデルを作成する予定です。

参考文献:
- [Analysis of Aggregated Bot and Human Traffic on E-Commerce Site](https://annals-csis.org/proceedings/2014/pliks/346.pdf)

## [WIP] 周期的な変動あるデータの時系列分析

[sample-time-series-analysis](https://github.com/kokeshiM0chi/demo-datascience/tree/main/sample-time-series-analysis)

- Prediction using wiki views

Kaggleコンペでは、Prophetを使用して、季節やイベントの影響を受けるデータの時系列分析を行ったので、その続編。項目数が多く、項目間に強い相関がある性質のデータセットに対する時系列分析も行う予定です。
手法の候補としてはiTransformerなどを想定しています。




