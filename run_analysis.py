"""
自動データ分析ツールの実行スクリプト
CSV/PDFファイルを読み込み、自動分析、可視化、予測を行う
"""

import os
import sys
import argparse
from auto_data_analysis import AutoDataAnalysis
from pdf_extractor import extract_main_table_from_pdf, check_pdf_libraries


def run_analysis(
    file_path,
    target_col=None,
    output_dir=None,
    api_key=None,
    drop_cols=None,
    date_cols=None,
):
    """
    ファイルを読み込んで分析を実行する

    Parameters:
    -----------
    file_path : str
        分析するファイルのパス（CSVまたはPDF）
    target_col : str, optional
        目的変数の列名
    output_dir : str, optional
        出力ディレクトリ (Noneの場合はダウンロードフォルダを使用)
    api_key : str, optional
        OpenAI APIキー
    drop_cols : list, optional
        除外する列名のリスト
    date_cols : list, optional
        日付として処理する列名のリスト

    Returns:
    --------
    AutoDataAnalysis
        分析インスタンス
    """
    # ファイル存在チェック
    if not os.path.exists(file_path):
        print(f"エラー: ファイル '{file_path}' が見つかりません")
        return None

    # ファイル拡張子を確認
    file_ext = os.path.splitext(file_path)[1].lower()

    # PDFファイルの場合、CSVに変換
    if file_ext == ".pdf":
        print("PDFファイルが検出されました - テーブルの抽出を試みます")

        if not check_pdf_libraries():
            print("PDFからデータを抽出するには追加ライブラリが必要です")
            print("pip install tabula-py pdfplumber PyPDF2")
            return None

        # PDFから主要なテーブルを抽出
        csv_path = extract_main_table_from_pdf(file_path, output_dir)

        if not csv_path:
            print("PDFからデータを抽出できませんでした。処理を中止します。")
            return None

        print(f"PDFからデータを抽出して一時ファイルに保存しました: {csv_path}")
        file_path = csv_path

    # 出力ディレクトリを設定
    if output_dir is None:
        output_dir = os.path.join(
            os.path.expanduser("~"), "Downloads", "data_analysis_results"
        )

    # OpenAI APIキーを環境変数から取得（指定がない場合）
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")

    print(f"分析結果の出力先: {output_dir}")

    # 自動分析インスタンスの作成
    analyzer = AutoDataAnalysis(api_key=api_key, output_dir=output_dir)

    try:
        # パイプラインを実行
        analyzer.run_pipeline(
            file_path=file_path,
            target_col=target_col,
            drop_cols=drop_cols,
            date_cols=date_cols,
        )

        print(
            f"分析が完了しました。レポートは {analyzer.report_file} に保存されています。"
        )

        # PDFから抽出した一時ファイルを削除
        if file_ext == ".pdf" and file_path != file_ext:
            try:
                os.remove(file_path)
                print(f"一時ファイルを削除しました: {file_path}")
            except:
                pass

        return analyzer

    except Exception as e:
        print(f"分析中にエラーが発生しました: {e}")
        return None


def main():
    """コマンドラインからの実行用メイン関数"""
    parser = argparse.ArgumentParser(description="自動データ分析ツール")
    parser.add_argument("file_path", help="分析するデータファイルのパス (CSV/PDF)")
    parser.add_argument("--target", help="目的変数の列名（予測対象）")
    parser.add_argument("--output", help="分析結果の出力ディレクトリ")
    parser.add_argument("--api_key", help="OpenAI APIキー")
    parser.add_argument("--drop", nargs="+", help="除外する列名（スペース区切り）")
    parser.add_argument(
        "--date", nargs="+", help="日付として処理する列名（スペース区切り）"
    )

    args = parser.parse_args()

    # 分析を実行
    analyzer = run_analysis(
        file_path=args.file_path,
        target_col=args.target,
        output_dir=args.output,
        api_key=args.api_key,
        drop_cols=args.drop,
        date_cols=args.date,
    )

    if analyzer:
        print("分析が正常に完了しました")
    else:
        print("分析が失敗しました")
        sys.exit(1)


if __name__ == "__main__":
    main()
