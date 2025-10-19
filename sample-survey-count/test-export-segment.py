import pandas as pd
import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt
from datetime import datetime

# -----------------
# 1. データの準備 (修正箇所)
# -----------------

# ユーザー指定のファイル名
FILE_NAME = 'count_20251018_221721.csv'

try:
    # CSVファイルの読み込み
    # 'date'列を日付型として解析し、インデックスに設定
    df = pd.read_csv(FILE_NAME, parse_dates=['date']).set_index('date')
except FileNotFoundError:
    print(f"エラー: ファイル '{FILE_NAME}' が見つかりません。")
    print("スクリプトと同じフォルダにファイルがあるか確認してください。")
    # 処理を中断
    exit()

# rupturesはnumpy配列を必要とするため、'event_count'列を抽出
signal = df['event_count'].values

# -----------------
# 2. 変化点検出
# -----------------

# Peltアルゴリズムを使用 (変化点の数が不明な場合に適している)
# model='l2' は平均（レベル）の変化を検出するのに適しています
# min_size は変化点間の最小データポイント数を指定 (任意で調整)
algo = rpt.Pelt(model="l2", min_size=1).fit(signal)

# 検出を実行
# penalty (ペナルティ) の値を調整して、検出される変化点の数を制御します。
# 適切な値はデータによって異なります。ここでは例として pen=100 を使用します。
# データの特徴に応じて、ペナルティ値を試行錯誤して調整してください。
penalty_value = 100 
result_indices = algo.predict(pen=penalty_value)

# rupturesの検出結果は変化後の最初のインデックスを返します。
# 最後の要素（データの長さ）を除外
change_point_indices = result_indices[:-1] 

# -----------------
# 3. セグメント境界の作成とCSV出力
# -----------------

# セグメントの開始と終了のリストを作成
segments = []

# 開始インデックスのリスト（0始まり + 各変化点）
start_indices = [0] + change_point_indices
# 終了インデックスのリスト（各変化点の1つ前 + 最後のデータ点）
end_indices = [idx - 1 for idx in change_point_indices] + [len(signal) - 1]

for i, (start_idx, end_idx) in enumerate(zip(start_indices, end_indices)):
    segment_data = signal[start_idx:end_idx+1]
    segments.append({
        'segment_id': i + 1,
        'start_date': df.index[start_idx].strftime('%Y-%m-%d'),
        'end_date': df.index[end_idx].strftime('%Y-%m-%d'),
        'start_index': start_idx,
        'end_index': end_idx,
        'duration_days': end_idx - start_idx + 1,
        'mean_event_count': np.mean(segment_data),
        'total_event_count': np.sum(segment_data)
    })

# DataFrameに変換
segments_df = pd.DataFrame(segments)

# CSV出力用のファイル名を生成
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_filename = f'segments_{timestamp}.csv'
segments_df.to_csv(output_filename, index=False, encoding='utf-8-sig')

print(f"\n📄 セグメント情報を '{output_filename}' に出力しました。")

# -----------------
# 4. 結果の表示と可視化
# -----------------

print(f"\n--- Ruptures 結果 ---")
print(f"対象ファイル: {FILE_NAME}")
print(f"アルゴリズム: Pelt (model='l2', penalty={penalty_value})")
print(f"検出されたセグメント数: {len(segments)}")
print(f"---------------------")

if len(change_point_indices) > 0:
    # 変化点のインデックスに対応する日付を取得
    change_point_dates = df.index[change_point_indices]
    
    print(f"\n✅ 変化が検出された日付 (変化後の最初の日):")
    for date in change_point_dates:
        print(f"- {date.strftime('%Y-%m-%d')}")
    
    print(f"\n📊 セグメント詳細:")
    for seg in segments:
        print(f"セグメント{seg['segment_id']}: {seg['start_date']} ～ {seg['end_date']} "
              f"({seg['duration_days']}日間, 平均={seg['mean_event_count']:.2f})")

    # 可視化
    plt.figure(figsize=(12, 6))
    
    # データをプロット
    plt.plot(signal, label='Event Count', linewidth=1)
    
    # 変化点を縦線で表示
    for cp in change_point_indices:
        plt.axvline(x=cp, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    plt.title(f"Change Point Detection on {FILE_NAME}")
    plt.xlabel("Data Index (日付順)")
    plt.ylabel("Event Count")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 日付ラベルをより見やすく表示
    tick_positions = np.arange(len(df))
    tick_labels = [date.strftime('%Y-%m-%d') for date in df.index]
    step = max(1, len(df) // 10) # 10個程度の間隔でラベルを表示
    plt.xticks(tick_positions[::step], tick_labels[::step], rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

else:
    print("⚠️ 検出された変化点はありませんでした。penaltyの値を小さくして再度お試しください。")
    # 変化点がない場合でも全体を1つのセグメントとして出力
    print(f"\n📄 セグメント情報（全期間）を '{output_filename}' に出力しました。")