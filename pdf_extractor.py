# pdf_extractor.py
"""
PDFからデータを抽出するモジュール
"""

import os
import pandas as pd
import tempfile
import warnings

warnings.filterwarnings("ignore")

# PDFライブラリが利用可能かチェックする
PDF_EXTRACTION_AVAILABLE = False
try:
    import tabula
    import pdfplumber

    PDF_EXTRACTION_AVAILABLE = True
except ImportError:
    pass


def check_pdf_libraries():
    """PDFを処理するライブラリが利用可能かチェックする"""
    if not PDF_EXTRACTION_AVAILABLE:
        print(
            "PDFデータ抽出機能を使用するには、追加ライブラリのインストールが必要です:"
        )
        print("pip install tabula-py pdfplumber PyPDF2")
        return False
    return True


def extract_tables_from_pdf(pdf_path):
    """
    PDFファイルからテーブルデータを抽出する

    Parameters:
    -----------
    pdf_path : str
        PDFファイルのパス

    Returns:
    --------
    list
        テーブルデータのリスト (pandas.DataFrame)
    """
    if not check_pdf_libraries():
        return None

    print(f"PDFからテーブルを抽出しています: {pdf_path}")

    # tabula-pyでテーブルを抽出
    try:
        tables = tabula.read_pdf(pdf_path, pages="all", multiple_tables=True)

        if tables and len(tables) > 0:
            print(f"{len(tables)}個のテーブルを抽出しました（tabula-py使用）")
            return tables
    except Exception as e:
        print(f"tabula-pyでの抽出中にエラーが発生しました: {e}")

    # tabula-pyで抽出できない場合はpdfplumberを試す
    print("pdfplumberでテーブルの抽出を試みます...")
    try:
        all_tables = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                tables = page.extract_tables()

                for j, table in enumerate(tables):
                    if table:
                        # ヘッダー行（最初の行）を取得
                        headers = table[0]

                        # ヘッダーが空の場合は列番号を使用
                        headers = [
                            h if h else f"Column_{i}" for i, h in enumerate(headers)
                        ]

                        # データ行（2行目以降）を取得
                        data = table[1:]

                        # DataFrameに変換
                        df = pd.DataFrame(data, columns=headers)
                        all_tables.append(df)

                        print(
                            f"ページ {i+1}, テーブル {j+1}: {df.shape[0]}行 x {df.shape[1]}列"
                        )

        if all_tables:
            print(f"{len(all_tables)}個のテーブルを抽出しました（pdfplumber使用）")
            return all_tables

    except Exception as e:
        print(f"pdfplumberでの抽出中にエラーが発生しました: {e}")

    print("PDFからテーブルを抽出できませんでした")
    return None


def extract_and_save_tables(pdf_path, output_dir=None):
    """
    PDFからテーブルを抽出してCSVファイルとして保存する

    Parameters:
    -----------
    pdf_path : str
        PDFファイルのパス
    output_dir : str, optional
        出力ディレクトリ（指定がない場合は一時ディレクトリを使用）

    Returns:
    --------
    list
        保存されたCSVファイルのパスのリスト
    """
    tables = extract_tables_from_pdf(pdf_path)

    if not tables:
        return None

    # 出力ディレクトリの設定
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    else:
        os.makedirs(output_dir, exist_ok=True)

    # PDFのベースネーム（拡張子なし）を取得
    pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]

    csv_paths = []
    for i, df in enumerate(tables):
        # NaNを含む列や行を削除（オプション）
        df_cleaned = df.dropna(how="all").dropna(axis=1, how="all")

        # CSVファイルとして保存
        csv_path = os.path.join(output_dir, f"{pdf_basename}_table_{i+1}.csv")
        df_cleaned.to_csv(csv_path, index=False)
        csv_paths.append(csv_path)

        print(f"テーブル {i+1} を保存しました: {csv_path}")

    return csv_paths


def select_main_table(csv_paths, min_rows=5):
    """
    複数のCSVファイルから最も重要なテーブルを選択する

    Parameters:
    -----------
    csv_paths : list
        CSVファイルのパスのリスト
    min_rows : int, default=5
        テーブルとみなす最小行数

    Returns:
    --------
    str
        選択されたCSVファイルのパス
    """
    if not csv_paths:
        return None

    # 単一のテーブルしかない場合はそれを返す
    if len(csv_paths) == 1:
        return csv_paths[0]

    # 各テーブルの情報を収集
    table_info = []
    for path in csv_paths:
        df = pd.read_csv(path)

        # 最小行数をチェック
        if df.shape[0] < min_rows:
            continue

        # テーブルの情報を記録
        info = {
            "path": path,
            "rows": df.shape[0],
            "cols": df.shape[1],
            "data_density": df.notna().sum().sum() / (df.shape[0] * df.shape[1]),
        }
        table_info.append(info)

    if not table_info:
        return csv_paths[0]  # 条件を満たすテーブルがない場合は最初のものを使用

    # データ密度と行数で並べ替え
    sorted_tables = sorted(
        table_info, key=lambda x: (x["data_density"], x["rows"]), reverse=True
    )

    # 最も重要なテーブルを返す
    return sorted_tables[0]["path"]


def extract_main_table_from_pdf(pdf_path, output_dir=None):
    """
    PDFから最も重要なテーブルを抽出してCSVとして保存する

    Parameters:
    -----------
    pdf_path : str
        PDFファイルのパス
    output_dir : str, optional
        出力ディレクトリ

    Returns:
    --------
    str
        抽出されたCSVファイルのパス、または失敗した場合はNone
    """
    # テーブルを抽出して保存
    csv_paths = extract_and_save_tables(pdf_path, output_dir)

    if not csv_paths:
        return None

    # 最も重要なテーブルを選択
    main_csv = select_main_table(csv_paths)

    # 最終的なCSVパスを返す
    return main_csv


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("使用方法: python pdf_extractor.py <PDFファイルパス> [出力ディレクトリ]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    main_csv = extract_main_table_from_pdf(pdf_path, output_dir)

    if main_csv:
        print(f"メインテーブルを抽出しました: {main_csv}")
    else:
        print("テーブルの抽出に失敗しました")
