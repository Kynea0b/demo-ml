import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_moons
import numpy as np


# 言語設定
## Macの標準フォント（ヒラギノ）を指定して日本語を有効化する
plt.rcParams['font.family'] = 'Hiragino Sans'

## マイナス記号が文字化けするのを防ぐ設定
plt.rcParams['axes.unicode_minus'] = False


# 1. 複雑な形状（三日月型）のデータセットに、あえてランダムなノイズ（外れ値）を混ぜる
X, _ = make_moons(n_samples=300, noise=0.08, random_state=42)
# ノイズとなるランダムな点をいくつか追加
np.random.seed(0)
noise = np.random.uniform(low=-1.0, high=2.0, size=(30, 2))
X = np.vstack([X, noise])

# 2. 比較用：K-means（クラスタ数を事前に「2」と指定して実行）
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X)

# 3. 本命：DBSCAN（密度の繋がりを見て自動でクラスタリング。ノイズは自動で「-1」になる）
dbscan = DBSCAN(eps=0.2, min_samples=5, metric='euclidean')
dbscan_labels = dbscan.fit_predict(X)

# 4. 結果をグラフで比較描画
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# K-meansの結果
axes[0].scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', s=50)
axes[0].set_title("K-means (形状を無視して直線の境界で強制分割してしまう)")

# DBSCANの結果（ノイズは灰色「-1」として綺麗に除外される）
axes[1].scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='tab10', s=50)
axes[1].set_title("DBSCAN (複雑な三日月型を捉え、ノイズを自動で除外)")

plt.tight_layout()
plt.show()

print(f"DBSCANが検出したクラスタ数 (ノイズ-1を除く): {len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)}")
print(f"DBSCANが検出しはじき出したノイズの数: (ラベル-1のデータ数): {np.sum(dbscan_labels == -1)}")
