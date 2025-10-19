import pandas as pd
import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt

# -----------------
# 1. データの準備 (修正箇所)
# -----------------

# ユーザー指定のファイル名
FILE_NAME = "count_20251018_221721.csv"

try:
    # CSVファイルの読み込み
    # 'date'列を日付型として解析し、インデックスに設定
    df = pd.read_csv(FILE_NAME, parse_dates=["date"]).set_index("date")
except FileNotFoundError:
    print(f"エラー: ファイル '{FILE_NAME}' が見つかりません。")
    print("スクリプトと同じフォルダにファイルがあるか確認してください。")
    # 処理を中断
    exit()

# rupturesはnumpy配列を必要とするため、'event_count'列を抽出
signal = df["event_count"].values

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
# 3. 結果の表示と可視化
# -----------------

print(f"--- Ruptures 結果 ---")
print(f"対象ファイル: {FILE_NAME}")
print(f"アルゴリズム: Pelt (model='l2', penalty={penalty_value})")
print(f"---------------------")

if len(change_point_indices) > 0:
    # 変化点のインデックスに対応する日付を取得
    change_point_dates = df.index[change_point_indices]

    print(f"✅ 変化が検出された日付 (変化後の最初の日):")
    for date in change_point_dates:
        print(f"- {date.strftime('%Y-%m-%d')}")

    # 可視化
    fig, axarr = rpt.display(
        signal, result_indices,
        computed_chg_pts_color="blue",
        computed_chg_pts_linewidth=3,
        computed_chg_pts_alpha=0.5,
        figsize=(12, 6)
    )

    # フォント設定（日本語対応）
    plt.rcParams['font.family'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo', 'sans-serif']

    # タイトルを図全体に設定
    fig.suptitle(f"Change Point Detection on {FILE_NAME}", fontsize=14)

    # 軸ラベルと目盛りの設定（axarrがリストでなくても動作する）
    # 最初に型をチェック
    if hasattr(axarr, 'set_xlabel'):  # 単一のAxesオブジェクトの場合
        ax = axarr
        ax.set_xlabel("Data Index (Date order)", fontsize=12)
        ax.set_ylabel("Event Count", fontsize=12)
        ax.grid(True)

        # 日付ラベルをより見やすく表示
        tick_positions = np.arange(len(df))
        tick_labels = [date.strftime("%Y-%m-%d") for date in df.index]
        step = max(1, len(df) // 10)  # 10個程度の間隔でラベルを表示
        ax.set_xticks(tick_positions[::step])
        ax.set_xticklabels(tick_labels[::step], rotation=45, ha="right")
    else:  # 複数のAxesオブジェクトがある場合
        # 最後のサブプロットにX軸ラベルを設定
        axarr[-1].set_xlabel("Data Index (Date order)", fontsize=12)

        # すべてのサブプロットにY軸ラベルとグリッドを設定
        for ax in axarr:
            ax.set_ylabel("Event Count", fontsize=12)
            ax.grid(True)

        # 最後のサブプロットに日付ラベルを設定
        tick_positions = np.arange(len(df))
        tick_labels = [date.strftime("%Y-%m-%d") for date in df.index]
        step = max(1, len(df) // 10)  # 10個程度の間隔でラベルを表示
        axarr[-1].set_xticks(tick_positions[::step])
        axarr[-1].set_xticklabels(tick_labels[::step], rotation=45, ha="right")

    plt.tight_layout()
    plt.show()

else:
    print(
        "⚠️ 検出された変化点はありませんでした。penaltyの値を小さくして再度お試しください。"
    )
