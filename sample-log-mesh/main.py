from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np

# 1. サンプルデータの用意（現場で起きうる様々なエラーログと、その時のユーザー行動）
# is_abandoned: True = 申請ボタンを押したのに失敗して離脱した（壊れたセッション）
raw_logs = [
    "Database connection timeout during submit",
    "Database connection timeout on query",       # 上とほぼ同じ意味（表現違い）
    "DB connection error: retry exhausted",     # 上に近い意味
    "Validation error: invalid email format",   # 軽微なバリデーションエラー（申請失敗とは直結しない）
    "Validation error: missing required field", # 軽微なバリデーションエラー
    "Unknown exception in profile service",     # 突然の致命的エラー
    "Unknown exception in profile service",     # 突然の致命的エラー
    "Background asset sync completed (info)",   # 単なる情報ログ（ノイズ）
    "Background asset sync completed (info)",   # 単なる情報ログ（ノイズ）
]

# 各ログに対応するユーザーの行動結果（このエラーを踏んだセッションが離脱したか）
user_abandoned_status = [
    True,  # Timeout -> 離脱
    True,  # Timeout -> 離脱
    True,  # Timeout -> 離脱
    False, # Validation -> 自力で直して進めるので離脱しない
    False, # Validation -> 離脱しない
    True,  # Unknown -> 離脱
    True,  # Unknown -> 離脱
    False, # Info -> 離脱しない
    False, # Info -> 離脱しない
]

df = pd.DataFrame({
    'log_message': raw_logs,
    'is_abandoned': user_abandoned_status
})

print("=== 1. 元のデータ（全9件） ===")
print(df[['log_message', 'is_abandoned']])
print("-" * 50)

# 2. 軽量な埋め込みモデル（Sentence Transformers）でテキストをベクトル化
# これにより「意味の近いエラー」がベクトル空間で近くに配置される
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['log_message'].tolist())

# 3. DBSCANでクラスタリング（コサイン類似度を使用）
# eps: 距離の閾値, min_samples: ひとつの塊とみなす最小数
dbscan = DBSCAN(eps=0.3, min_samples=2, metric='cosine')
clusters = dbscan.fit_predict(embeddings)

df['error_cluster'] = clusters
print("=== 2. DBSCANによるクラスタリング結果 (-1はノイズ) ===")
print(df[['error_cluster', 'log_message']])
print("-" * 50)

# 4. ユーザー行動（離脱）と交差させて、クラスターごとの「影響度（F1スコア）」を評価
impact_results = []
for cluster_id in sorted(df['error_cluster'].unique()):
    if cluster_id == -1:
        cluster_name = "Noise (ノイズ・孤立ログ)"
    else:
        cluster_name = f"Cluster_{cluster_id}"

    group = df[df['error_cluster'] == cluster_id]

    # 予測：このエラークラスターが発生したセッションは「申請失敗（True）」とみなす
    y_pred = [True] * len(group)
    # 実態：実際にユーザーが離脱したか
    y_true = group['is_abandoned'].tolist()

    # F1スコアを算出（このエラーが出たとき、どれくらい確実にユーザーを止めているか）
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # 離脱率の計算を事前に分ける
    mean_abandon = sum(y_true) / len(y_true)

    impact_results.append({
        'Cluster': cluster_name,
        'Sample Count': len(group),
        'Actual Abandon Rate': mean_abandon,
        'Impact (F1 Score)': f1,
        'Representative Log': group['log_message'].iloc[0]
    })

result_df = pd.DataFrame(impact_results)
print("=== 3. 最終的な影響度評価（SLI/アラートの判断材料） ===")
print(result_df[['Cluster', 'Sample Count', 'Impact (F1 Score)', 'Representative Log']])
