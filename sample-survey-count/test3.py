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
# 4. 結果の表示
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

# -----------------
# 5. 可視化（上段：元データ、下段：セグメント表示）
# -----------------

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

# --- 上段: 元の変化点検出グラフ ---
ax1.plot(signal, label='Event Count', linewidth=1.5, color='steelblue', zorder=3)

# 変化点を縦線で表示
for cp in change_point_indices:
    ax1.axvline(x=cp, color='red', linestyle='--', linewidth=2, alpha=0.7, zorder=2)

ax1.set_title(f"Original Data with Change Points - {FILE_NAME}", fontsize=14, fontweight='bold')
ax1.set_xlabel("Data Index", fontsize=11)
ax1.set_ylabel("Event Count", fontsize=11)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 日付ラベル
tick_positions = np.arange(len(df))
tick_labels = [date.strftime('%m/%d') for date in df.index]
step = max(1, len(df) // 15)
ax1.set_xticks(tick_positions[::step])
ax1.set_xticklabels(tick_labels[::step], rotation=45, ha='right', fontsize=9)

# --- 下段: セグメント区切りを強調（縦線なし） ---
colors = plt.cm.Set3(np.linspace(0, 1, len(segments)))

for i, seg in enumerate(segments):
    start_idx = seg['start_index']
    end_idx = seg['end_index']
    mean_val = seg['mean_event_count']
    
    # セグメントごとに背景色を塗りつぶし（これが区切りを表現）
    ax2.axvspan(start_idx, end_idx, alpha=0.3, color=colors[i], zorder=1)
    
    # 実データをプロット
    segment_x = range(start_idx, end_idx+1)
    segment_y = signal[start_idx:end_idx+1]
    ax2.plot(segment_x, segment_y, color='black', linewidth=2, zorder=3)
    
    # セグメントの平均値を水平線で表示
    ax2.hlines(y=mean_val, xmin=start_idx, xmax=end_idx, 
               colors='red', linewidth=2.5, linestyle='-', alpha=0.7, zorder=2,
               label=f'Seg{seg["segment_id"]} 平均' if i < 5 else '')

ax2.set_title("Segment Boundaries (背景色でセグメント区切りを表示)", fontsize=14, fontweight='bold')
ax2.set_xlabel("Data Index", fontsize=11)
ax2.set_ylabel("Event Count", fontsize=11)
ax2.grid(True, alpha=0.3, axis='y', zorder=0)

# 凡例は最初の5セグメントのみ表示（多すぎる場合の対策）
if len(segments) <= 5:
    ax2.legend(loc='upper left', fontsize=9)

# 日付ラベル
ax2.set_xticks(tick_positions[::step])
ax2.set_xticklabels(tick_labels[::step], rotation=45, ha='right', fontsize=9)

# セグメント情報を図の下に追加
segment_info_text = " | ".join([f"S{seg['segment_id']}:{seg['start_date']}～{seg['end_date']}(平均{seg['mean_event_count']:.1f})" 
                                 for seg in segments])
fig.text(0.5, 0.01, segment_info_text, ha='center', fontsize=8, 
         wrap=True, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.show()

if len(change_point_indices) == 0:
    print("⚠️ 検出された変化点はありませんでした。penaltyの値を小さくして再度お試しください。")