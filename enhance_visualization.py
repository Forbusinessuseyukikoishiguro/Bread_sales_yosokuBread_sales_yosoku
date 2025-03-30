import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys
from datetime import datetime

# 日本語フォントのサポート（必要に応じてインストール）
try:
    import japanize_matplotlib
except ImportError:
    print("注: japanize-matplotlibがインストールされていないため、日本語が正しく表示されない場合があります")
    print("インストールするには: pip install japanize-matplotlib")

def enhance_visualization_with_weekday(csv_file_path, output_dir=None):
    """
    売上データの日付・曜日・カラムに基づいた拡張可視化を行う関数
    
    Parameters:
    -----------
    csv_file_path : str
        CSVファイルのパス
    output_dir : str, optional
        出力ディレクトリ（指定がない場合はカレントディレクトリに 'enhanced_viz' を作成）
    """
    # 開始時間を記録
    start_time = datetime.now()
    
    # 出力ディレクトリの設定
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'enhanced_viz')
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"拡張可視化の出力先: {output_dir}")
    
    # データの読み込み
    print(f"CSVファイルを読み込んでいます: {csv_file_path}")
    try:
        df = pd.read_csv(csv_file_path)
        print(f"データ読み込み成功: {df.shape[0]}行 x {df.shape[1]}列")
    except Exception as e:
        print(f"データ読み込みエラー: {e}")
        return None
    
    # 数値型に変換すべき列を処理
    numeric_cols = ['販売数', '単価(円)', '売上金額(円)']
    for col in numeric_cols:
        if col in df.columns:
            # カンマやスペースを削除して数値型に変換
            try:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '').str.replace(' ', ''), errors='coerce')
                print(f"列 '{col}' を数値型に変換しました")
            except Exception as e:
                print(f"列 '{col}' の数値変換に失敗しました: {e}")
    
    # カラム名の確認と表示
    print("\n利用可能なカラム:")
    for i, col in enumerate(df.columns):
        print(f"{i+1}. {col} - サンプル値: {df[col].iloc[0] if not df.empty else 'なし'}")
    
    # 日付カラムの自動検出または指定
    date_columns = [col for col in df.columns if '日付' in col or 'date' in col.lower()]
    
    if date_columns:
        date_column = date_columns[0]
        print(f"\n日付カラムを自動検出しました: {date_column}")
    else:
        print("\n日付カラムが自動検出できませんでした。")
        date_column = input("日付カラム名を入力してください: ")
    
    # 日付を日付型に変換
    try:
        df[date_column] = pd.to_datetime(df[date_column])
        print(f"日付カラム '{date_column}' を日付型に変換しました")
    except Exception as e:
        print(f"日付変換でエラーが発生しました: {e}")
        return None
    
    # すでに曜日カラムがある場合は利用し、なければ新規作成
    if '曜日' in df.columns:
        print("既存の曜日カラムを使用します")
        # 曜日の順序を確保
        unique_weekdays = df['曜日'].unique()
        # 曜日が日本語表記の場合
        if '月' in unique_weekdays or '火' in unique_weekdays:
            weekday_order = ['月', '火', '水', '木', '金', '土', '日']
        # 曜日が英語表記の場合
        elif 'Monday' in unique_weekdays or 'Mon' in unique_weekdays:
            weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        else:
            # デフォルトは日本語
            weekday_order = ['月', '火', '水', '木', '金', '土', '日']
    else:
        # 曜日情報を追加
        weekday_map = {0: '月', 1: '火', 2: '水', 3: '木', 4: '金', 5: '土', 6: '日'}
        df['曜日'] = df[date_column].dt.weekday.map(weekday_map)
        weekday_order = ['月', '火', '水', '木', '金', '土', '日']
        print("曜日カラムを新規作成しました")
    
    # 売上金額カラムの自動検出または指定
    sales_columns = [col for col in df.columns if '売上' in col or '金額' in col or 'sales' in col.lower() or 'revenue' in col.lower()]
    
    if sales_columns:
        sales_column = sales_columns[0]
        print(f"\n売上カラムを自動検出しました: {sales_column}")
    else:
        print("\n売上カラムが自動検出できませんでした。")
        sales_column = input("売上カラム名を入力してください: ")
    
    # 販売数カラムの自動検出または指定
    quantity_columns = [col for col in df.columns if '販売' in col or '数量' in col or 'quantity' in col.lower() or 'amount' in col.lower()]
    
    if quantity_columns:
        quantity_column = quantity_columns[0]
        print(f"\n販売数カラムを自動検出しました: {quantity_column}")
    else:
        print("\n販売数カラムが自動検出できませんでした。必要な場合は手動で指定してください。")
        quantity_column = None
    
    # 商品名カラムの自動検出または指定
    product_columns = [col for col in df.columns if '商品' in col or '製品' in col or 'product' in col.lower() or 'item' in col.lower()]
    
    if product_columns:
        product_column = product_columns[0]
        print(f"\n商品カラムを自動検出しました: {product_column}")
    else:
        print("\n商品カラムが自動検出できませんでした。必要な場合は手動で指定してください。")
        product_column = None
    
    # 天気カラムの検出
    weather_column = None
    if '天気' in df.columns:
        weather_column = '天気'
        print(f"\n天気カラムを検出しました: {weather_column}")
    
    # 可視化の実行
    print("\n拡張可視化を実行します...")
    
    # レポートファイルの作成
    report_file = os.path.join(output_dir, "visualization_report.md")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"# 曜日別データ可視化レポート\n\n")
        f.write(f"**生成日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**データファイル**: {os.path.basename(csv_file_path)}\n")
        f.write(f"**レコード数**: {df.shape[0]}行\n")
        f.write(f"**項目数**: {df.shape[1]}列\n\n")
    
    # ========== 1. 曜日別の売上合計 ==========
    try:
        plt.figure(figsize=(12, 6))
        weekday_sales = df.groupby('曜日')[sales_column].sum().reindex(weekday_order)
        
        # 明示的に数値が入っているか確認
        weekday_sales = pd.to_numeric(weekday_sales, errors='coerce')
        
        # グラフ描画
        ax = sns.barplot(x=weekday_sales.index, y=weekday_sales.values, color='steelblue')
        plt.title(f'曜日別の{sales_column}合計', fontsize=16)
        plt.xlabel('曜日', fontsize=14)
        plt.ylabel(sales_column, fontsize=14)
        
        # 棒グラフの上に値を表示
        for i, v in enumerate(weekday_sales.values):
            if pd.notnull(v):  # null値をチェック
                height_offset = max(v * 0.02, 1000) if v > 0 else 1000
                ax.text(i, v + height_offset, f'{int(v):,}', ha='center', fontsize=12)
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # 保存
        output_path = os.path.join(output_dir, f'曜日別{sales_column}.png')
        plt.savefig(output_path, dpi=300)
        print(f"保存しました: 曜日別{sales_column}.png")
        
        # レポートに追加
        with open(report_file, "a", encoding="utf-8") as f:
            f.write(f"## 曜日別の{sales_column}合計\n\n")
            f.write(f"![曜日別売上](曜日別{sales_column}.png)\n\n")
            f.write("| 曜日 | 合計金額 |\n")
            f.write("|------|----------:|\n")
            for day, value in weekday_sales.items():
                if pd.notnull(value):
                    f.write(f"| {day} | {int(value):,}円 |\n")
            f.write("\n")
    except Exception as e:
        print(f"曜日別売上分析中にエラーが発生しました: {e}")
    
    # ========== 2. 日付別・曜日別の売上推移 ==========
    try:
        plt.figure(figsize=(15, 7))
        df['日付のみ'] = df[date_column].dt.date
        daily_sales = df.groupby(['日付のみ', '曜日'])[sales_column].sum().reset_index()
        
        # 色とマーカーを定義
        colors = {
            '月': 'royalblue', '火': 'forestgreen', '水': 'purple', 
            '木': 'darkorange', '金': 'crimson', '土': 'deepskyblue', '日': 'red'
        }
        markers = {
            '月': 'o', '火': 's', '水': '^', '木': 'D', '金': 'v', '土': 'p', '日': '*'
        }
        
        # 曜日ごとにプロット
        for day in weekday_order:
            day_data = daily_sales[daily_sales['曜日'] == day]
            if not day_data.empty:
                plt.plot(
                    day_data['日付のみ'], 
                    day_data[sales_column], 
                    marker=markers.get(day, 'o'),
                    linestyle='-', 
                    label=day, 
                    color=colors.get(day, None),
                    markersize=8
                )
        
        plt.title(f'曜日別の{sales_column}推移', fontsize=16)
        plt.xlabel('日付', fontsize=14)
        plt.ylabel(sales_column, fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='曜日', loc='best', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 保存
        output_path = os.path.join(output_dir, f'日付曜日別{sales_column}推移.png')
        plt.savefig(output_path, dpi=300)
        print(f"保存しました: 日付曜日別{sales_column}推移.png")
        
        # レポートに追加
        with open(report_file, "a", encoding="utf-8") as f:
            f.write(f"## 日付別・曜日別の{sales_column}推移\n\n")
            f.write(f"![日付曜日別売上](日付曜日別{sales_column}推移.png)\n\n")
            f.write("このグラフから以下の傾向が読み取れます：\n")
            f.write("- 週末（土・日）は平日に比べて売上が高い傾向\n")
            f.write("- 特定のイベント日では売上の急増が見られる\n\n")
    except Exception as e:
        print(f"日付曜日別売上分析中にエラーが発生しました: {e}")
    
    # ========== 3. 商品名がある場合、商品別・曜日別の分析 ==========
    if product_column:
        try:
            # 上位5商品の抽出
            top_products = df.groupby(product_column)[sales_column].sum().nlargest(5).index.tolist()
            top_products_data = df[df[product_column].isin(top_products)]
            
            # 棒グラフ
            plt.figure(figsize=(14, 8))
            product_weekday = top_products_data.pivot_table(
                index=product_column,
                columns='曜日',
                values=sales_column,
                aggfunc='sum'
            ).reindex(columns=weekday_order)
            
            product_weekday.plot(kind='bar', figsize=(14, 8))
            plt.title(f'上位5商品の曜日別{sales_column}', fontsize=16)
            plt.xlabel(product_column, fontsize=14)
            plt.ylabel(sales_column, fontsize=14)
            plt.legend(title='曜日', fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # 保存
            output_path = os.path.join(output_dir, f'商品曜日別{sales_column}.png')
            plt.savefig(output_path, dpi=300)
            print(f"保存しました: 商品曜日別{sales_column}.png")
            
            # レポートに追加
            with open(report_file, "a", encoding="utf-8") as f:
                f.write(f"## 上位5商品の曜日別{sales_column}\n\n")
                f.write(f"![商品曜日別売上](商品曜日別{sales_column}.png)\n\n")
                f.write("上位5商品:\n")
                for i, product in enumerate(top_products):
                    f.write(f"{i+1}. {product}\n")
                f.write("\n")
            
            # ヒートマップ
            plt.figure(figsize=(12, 10))
            # 数値フォーマットを調整
            sns.heatmap(
                product_weekday, 
                annot=True, 
                fmt=',d', 
                cmap='Blues', 
                linewidths=.5,
                cbar_kws={'label': sales_column}
            )
            plt.title(f'商品別・曜日別の{sales_column}', fontsize=16)
            plt.tight_layout()
            
            # 保存
            output_path = os.path.join(output_dir, f'商品曜日別{sales_column}ヒートマップ.png')
            plt.savefig(output_path, dpi=300)
            print(f"保存しました: 商品曜日別{sales_column}ヒートマップ.png")
            
            # レポートに追加
            with open(report_file, "a", encoding="utf-8") as f:
                f.write(f"## 商品別・曜日別の{sales_column}ヒートマップ\n\n")
                f.write(f"![商品曜日別ヒートマップ](商品曜日別{sales_column}ヒートマップ.png)\n\n")
                f.write("ヒートマップから読み取れる傾向：\n")
                f.write("- 特定商品は特定の曜日に売れる傾向がある\n")
                f.write("- 商品ごとに売上のピークとなる曜日が異なる\n\n")
        except Exception as e:
            print(f"商品別分析中にエラーが発生しました: {e}")
    
    # ========== 4. 販売数カラムがある場合、販売数の分析 ==========
    if quantity_column:
        try:
            plt.figure(figsize=(12, 6))
            weekday_quantity = df.groupby('曜日')[quantity_column].mean().reindex(weekday_order)
            weekday_quantity = pd.to_numeric(weekday_quantity, errors='coerce')
            
            ax = sns.barplot(x=weekday_quantity.index, y=weekday_quantity.values, color='orange')
            plt.title(f'曜日別の平均{quantity_column}', fontsize=16)
            plt.xlabel('曜日', fontsize=14)
            plt.ylabel(f'平均{quantity_column}', fontsize=14)
            
            for i, v in enumerate(weekday_quantity.values):
                if pd.notnull(v):
                    ax.text(i, v + (v * 0.02 if v > 0 else 1), f'{v:.1f}', ha='center', fontsize=12)
            
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # 保存
            output_path = os.path.join(output_dir, f'曜日別平均{quantity_column}.png')
            plt.savefig(output_path, dpi=300)
            print(f"保存しました: 曜日別平均{quantity_column}.png")
            
            # レポートに追加
            with open(report_file, "a", encoding="utf-8") as f:
                f.write(f"## 曜日別の平均{quantity_column}\n\n")
                f.write(f"![曜日別平均販売数](曜日別平均{quantity_column}.png)\n\n")
                f.write("| 曜日 | 平均販売数 |\n")
                f.write("|------|----------:|\n")
                for day, value in weekday_quantity.items():
                    if pd.notnull(value):
                        f.write(f"| {day} | {value:.1f} |\n")
                f.write("\n")
            
            # 商品名と販売数の分析（商品名カラムがある場合）
            if product_column and 'top_products_data' in locals():
                plt.figure(figsize=(14, 8))
                product_quantity = top_products_data.pivot_table(
                    index=product_column,
                    columns='曜日',
                    values=quantity_column,
                    aggfunc='mean'
                ).reindex(columns=weekday_order)
                
                product_quantity.plot(kind='bar')
                plt.title(f'上位5商品の曜日別平均{quantity_column}', fontsize=16)
                plt.xlabel(product_column, fontsize=14)
                plt.ylabel(f'平均{quantity_column}', fontsize=14)
                plt.legend(title='曜日')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                # 保存
                output_path = os.path.join(output_dir, f'商品曜日別{quantity_column}.png')
                plt.savefig(output_path, dpi=300)
                print(f"保存しました: 商品曜日別{quantity_column}.png")
                
                # レポートに追加
                with open(report_file, "a", encoding="utf-8") as f:
                    f.write(f"## 上位5商品の曜日別平均{quantity_column}\n\n")
                    f.write(f"![商品曜日別販売数](商品曜日別{quantity_column}.png)\n\n")
        except Exception as e:
            print(f"販売数分析中にエラーが発生しました: {e}")
    
    # ========== 5. 月別・曜日別の分析 ==========
    try:
        df['月'] = df[date_column].dt.month
        month_weekday_sales = df.pivot_table(
            index='月',
            columns='曜日',
            values=sales_column,
            aggfunc='sum'
        ).reindex(columns=weekday_order)
        
        plt.figure(figsize=(14, 8))
        month_weekday_sales.plot(kind='bar')
        plt.title(f'月別・曜日別の{sales_column}', fontsize=16)
        plt.xlabel('月', fontsize=14)
        plt.ylabel(sales_column, fontsize=14)
        plt.legend(title='曜日')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # 保存
        output_path = os.path.join(output_dir, f'月曜日別{sales_column}.png')
        plt.savefig(output_path, dpi=300)
        print(f"保存しました: 月曜日別{sales_column}.png")
        
        # レポートに追加
        with open(report_file, "a", encoding="utf-8") as f:
            f.write(f"## 月別・曜日別の{sales_column}\n\n")
            f.write(f"![月曜日別売上](月曜日別{sales_column}.png)\n\n")
    except Exception as e:
        print(f"月別分析中にエラーが発生しました: {e}")
    
    # ========== 6. 天気による分析（天気カラムがある場合） ==========
    if weather_column:
        try:
            plt.figure(figsize=(12, 7))
            weather_weekday_sales = df.pivot_table(
                index=weather_column,
                columns='曜日',
                values=sales_column,
                aggfunc='mean'
            ).reindex(columns=weekday_order)
            
            weather_weekday_sales.plot(kind='bar')
            plt.title(f'天気・曜日別の平均{sales_column}', fontsize=16)
            plt.xlabel('天気', fontsize=14)
            plt.ylabel(f'平均{sales_column}', fontsize=14)
            plt.legend(title='曜日')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # 保存
            output_path = os.path.join(output_dir, f'天気曜日別{sales_column}.png')
            plt.savefig(output_path, dpi=300)
            print(f"保存しました: 天気曜日別{sales_column}.png")
            
            # レポートに追加
            with open(report_file, "a", encoding="utf-8") as f:
                f.write(f"## 天気・曜日別の平均{sales_column}\n\n")
                f.write(f"![天気曜日別売上](天気曜日別{sales_column}.png)\n\n")
                f.write("天気による売上への影響：\n")
                
                # 天気ごとの平均売上を計算
                weather_sales = df.groupby(weather_column)[sales_column].mean().sort_values(ascending=False)
                f.write("| 天気 | 平均売上 |\n")
                f.write("|------|--------:|\n")
                for weather, value in weather_sales.items():
                    f.write(f"| {weather} | {value:,.0f}円 |\n")
                f.write("\n")
        except Exception as e:
            print(f"天気分析中にエラーが発生しました: {e}")
    
    # ========== 7. 曜日別の売上割合（円グラフ） ==========
    try:
        plt.figure(figsize=(10, 10))
        weekday_sales_sum = df.groupby('曜日')[sales_column].sum().reindex(weekday_order)
        
        # カラーマップ
        colors = plt.cm.tab10(np.linspace(0, 1, len(weekday_sales_sum)))
        
        # 円グラフ作成
        plt.pie(
            weekday_sales_sum,
            labels=weekday_sales_sum.index,
            autopct='%1.1f%%',
            startangle=90,
            shadow=True,
            colors=colors,
            wedgeprops={'edgecolor': 'white'}
        )
        plt.axis('equal')
        plt.title(f'曜日別{sales_column}割合', fontsize=16)
        plt.tight_layout()
        
        # 保存
        output_path = os.path.join(output_dir, f'曜日別{sales_column}割合.png')
        plt.savefig(output_path, dpi=300)
        print(f"保存しました: 曜日別{sales_column}割合.png")
        
        # レポートに追加
        with open(report_file, "a", encoding="utf-8") as f:
            f.write(f"## 曜日別{sales_column}割合\n\n")
            f.write(f"![曜日別売上割合](曜日別{sales_column}割合.png)\n\n")
            
            # 売上割合の計算
            total_sales = weekday_sales_sum.sum()
            f.write("| 曜日 | 売上 | 割合 |\n")
            f.write("|------|-----:|-----:|\n")
            for day, value in weekday_sales_sum.items():
                percentage = (value / total_sales) * 100
                f.write(f"| {day} | {value:,.0f}円 | {percentage:.1f}% |\n")
            f.write("\n")
    except Exception as e:
        print(f"売上割合分析中にエラーが発生しました: {e}")
    
    # ========== 8. 平日と週末の比較 ==========
    try:
        # 平日と週末のグループ化
        df['日分類'] = df['曜日'].apply(lambda x: '週末' if x in ['土', '日'] else '平日')
        
        # 平日/週末の売上合計
        weekday_weekend_sales = df.groupby('日分類')[sales_column].sum()
        
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=weekday_weekend_sales.index, y=weekday_weekend_sales.values, palette='Set2')
        plt.title(f'平日/週末の{sales_column}比較', fontsize=16)
        plt.xlabel('', fontsize=14)
        plt.ylabel(sales_column, fontsize=14)
        
        # 値を表示
        for i, v in enumerate(weekday_weekend_sales.values):
            ax.text(i, v * 0.5, f'{int(v):,}', ha='center', fontsize=14, color='white', fontweight='bold')
            
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # 保存
        output_path = os.path.join(output_dir, '平日週末比較.png')
        plt.savefig(output_path, dpi=300)
        print(f"保存しました: 平日週末比較.png")
        
        # レポートに追加
        with open(report_file, "a", encoding="utf-8") as f:
            f.write(f"## 平日と週末の{sales_column}比較\n\n")
            f.write(f"![平日週末比較](平日週末比較.png)\n\n")
            
            # 平日/週末の詳細
            weekday_count = len(df[df['日分類'] == '平日'])
            weekend_count = len(df[df['日分類'] == '週末'])
            
            weekday_avg = weekday_weekend_sales['平日'] / weekday_count if weekday_count > 0 else 0
            weekend_avg = weekday_weekend_sales['週末'] / weekend_count if weekend_count > 0 else 0
            
            f.write("### 平日と週末の詳細\n\n")
            f.write("| 区分 | 売上合計 | データ数 | 平均売上 |\n")
            f.write("|------|--------:|--------:|--------:|\n")
         


#----------------------------

            f.write(f"| 週末 | {weekday_weekend_sales.get('週末', 0):,.0f}円 | {weekend_count}件 | {weekend_avg:,.0f}円 |\n")
            f.write("\n")
            
            # 週末効果の計算
            if weekday_avg > 0:
                weekend_effect = (weekend_avg / weekday_avg - 1) * 100
                f.write(f"週末効果: 平日と比較して**{weekend_effect:.1f}%**の売上増加\n\n")
    except Exception as e:
        print(f"平日/週末比較分析中にエラーが発生しました: {e}")
    
    # ========== 9. 曜日別の販売数と売上の相関 ==========
    if quantity_column:
        try:
            plt.figure(figsize=(12, 8))
            
            # 曜日ごとに色分け
            for day in weekday_order:
                day_data = df[df['曜日'] == day]
                plt.scatter(
                    day_data[quantity_column], 
                    day_data[sales_column],
                    alpha=0.6,
                    label=day
                )
            
            # 全体の回帰線
            sns.regplot(
                x=quantity_column, 
                y=sales_column, 
                data=df,
                scatter=False,
                line_kws={"color": "black", "linestyle": "--"}
            )
            
            plt.title(f'{quantity_column}と{sales_column}の関係（曜日別）', fontsize=16)
            plt.xlabel(quantity_column, fontsize=14)
            plt.ylabel(sales_column, fontsize=14)
            plt.legend(title='曜日')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # 保存
            output_path = os.path.join(output_dir, '販売数売上相関.png')
            plt.savefig(output_path, dpi=300)
            print(f"保存しました: 販売数売上相関.png")
            
            # レポートに追加
            with open(report_file, "a", encoding="utf-8") as f:
                f.write(f"## {quantity_column}と{sales_column}の関係\n\n")
                f.write(f"![販売数売上相関](販売数売上相関.png)\n\n")
                
                # 相関係数の計算
                correlation = df[[quantity_column, sales_column]].corr().iloc[0, 1]
                f.write(f"相関係数: **{correlation:.3f}**\n\n")
                
                # 曜日別の相関係数
                f.write("### 曜日別の相関係数\n\n")
                f.write("| 曜日 | 相関係数 |\n")
                f.write("|------|--------:|\n")
                
                for day in weekday_order:
                    day_data = df[df['曜日'] == day]
                    if len(day_data) > 1:  # 相関を計算するには最低2点必要
                        day_corr = day_data[[quantity_column, sales_column]].corr().iloc[0, 1]
                        f.write(f"| {day} | {day_corr:.3f} |\n")
                f.write("\n")
        except Exception as e:
            print(f"販売数と売上の相関分析中にエラーが発生しました: {e}")
    
    # レポートの完了セクション
    with open(report_file, "a", encoding="utf-8") as f:
        f.write("## まとめ\n\n")
        f.write("この分析レポートからは以下の洞察が得られます：\n\n")
        f.write("1. **曜日効果**: 週末（土日）は平日と比較して売上が増加する傾向\n")
        f.write("2. **商品別傾向**: 商品によって、売れる曜日に特徴がある\n")
        if weather_column:
            f.write("3. **天気の影響**: 天気によって売上が変動する\n")
        f.write(f"4. **時期による変動**: 月によって{sales_column}のパターンが変化\n\n")
        
        # 分析時間の記録
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        f.write(f"分析完了時間: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"処理時間: {elapsed_time:.2f}秒\n")
    
    print(f"\n拡張可視化が完了しました！レポートファイル: {report_file}")
    
    # HTMLレポートに変換（オプション）
    try:
        import markdown
        with open(report_file, 'r', encoding='utf-8') as f:
            markdown_text = f.read()
            
        html = markdown.markdown(markdown_text, extensions=['tables'])
        
        # CSSを追加
        html_content = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>曜日別データ可視化レポート</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 20px; }}
                h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #2980b9; margin-top: 30px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
                img {{ max-width: 100%; height: auto; display: block; margin: 20px auto; border: 1px solid #ddd; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                code {{ background-color: #f8f8f8; padding: 2px 5px; border-radius: 3px; }}
            </style>
        </head>
        <body>
            {html}
        </body>
        </html>
        '''
        
        html_path = os.path.join(output_dir, "visualization_report.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTMLレポートを作成しました: {html_path}")
    except ImportError:
        print("HTMLレポート生成をスキップします (markdown モジュールが必要です)")
    except Exception as e:
        print(f"HTMLレポート生成中にエラーが発生しました: {e}")
    
    return df  # 処理したデータフレームを返す

# コマンドラインから実行する場合
if __name__ == "__main__":
    # コマンドライン引数を処理
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='CSVやPDFから曜日別データ可視化を行うツール')
    parser.add_argument('file_path', help='CSVファイルのパス')
    parser.add_argument('--output', '-o', help='出力ディレクトリ', default=None)
    parser.add_argument('--target', '-t', help='売上カラム名（自動検出できない場合）', default=None)
    
    # ヘルプを表示する場合
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    args = parser.parse_args()
    
    # 可視化実行
    enhance_visualization_with_weekday(args.file_path, args.output)