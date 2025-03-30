"""
自動データ分析ツールのGUIインターフェース
CSVやPDFファイルを読み込んで自動分析、可視化を行う
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import webbrowser
import subprocess
from auto_data_analysis import AutoDataAnalysis
from pdf_extractor import extract_main_table_from_pdf, check_pdf_libraries
from run_analysis import run_analysis


class AutoDataAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("自動データ分析ツール")
        self.root.geometry("600x650")
        self.root.resizable(True, True)

        # ダウンロードフォルダのパスを取得
        self.download_dir = os.path.join(
            os.path.expanduser("~"), "Downloads", "data_analysis_results"
        )

        # 出力ディレクトリを作成
        os.makedirs(self.download_dir, exist_ok=True)

        # 分析タスク管理用の変数
        self.current_task = None
        self.analyzer = None

        # GUIの作成
        self.create_widgets()

    def create_widgets(self):
        """GUIウィジェットの作成"""
        # メインフレーム
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # タイトルラベル
        title_label = ttk.Label(
            main_frame, text="自動データ分析ツール", font=("Arial", 16, "bold")
        )
        title_label.pack(pady=(0, 20))

        # ファイル選択セクション
        file_frame = ttk.LabelFrame(main_frame, text="ファイル選択", padding=10)
        file_frame.pack(fill=tk.X, pady=10)

        file_select_frame = ttk.Frame(file_frame)
        file_select_frame.pack(fill=tk.X, pady=5)

        self.file_entry = ttk.Entry(file_select_frame)
        self.file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        browse_btn = ttk.Button(
            file_select_frame, text="参照...", command=self.browse_file
        )
        browse_btn.pack(side=tk.RIGHT)

        # 分析設定セクション
        settings_frame = ttk.LabelFrame(main_frame, text="分析設定", padding=10)
        settings_frame.pack(fill=tk.X, pady=10)

        # 目的変数設定
        target_frame = ttk.Frame(settings_frame)
        target_frame.pack(fill=tk.X, pady=5)

        ttk.Label(target_frame, text="目的変数（予測対象）:").pack(side=tk.LEFT)
        self.target_entry = ttk.Entry(target_frame)
        self.target_entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        # 日付列設定
        date_frame = ttk.Frame(settings_frame)
        date_frame.pack(fill=tk.X, pady=5)

        ttk.Label(date_frame, text="日付列（カンマ区切り）:").pack(side=tk.LEFT)
        self.date_entry = ttk.Entry(date_frame)
        self.date_entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        # 除外列設定
        drop_frame = ttk.Frame(settings_frame)
        drop_frame.pack(fill=tk.X, pady=5)

        ttk.Label(drop_frame, text="除外列（カンマ区切り）:").pack(side=tk.LEFT)
        self.drop_entry = ttk.Entry(drop_frame)
        self.drop_entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        # APIキー設定
        api_frame = ttk.Frame(settings_frame)
        api_frame.pack(fill=tk.X, pady=5)

        ttk.Label(api_frame, text="OpenAI APIキー（任意）:").pack(side=tk.LEFT)
        self.api_entry = ttk.Entry(api_frame, show="*")
        self.api_entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        # 出力ディレクトリ設定
        output_frame = ttk.Frame(settings_frame)
        output_frame.pack(fill=tk.X, pady=5)

        ttk.Label(output_frame, text="出力ディレクトリ:").pack(side=tk.LEFT)
        self.output_entry = ttk.Entry(output_frame)
        self.output_entry.insert(0, self.download_dir)
        self.output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        output_browse_btn = ttk.Button(
            output_frame, text="参照...", command=self.browse_output_dir
        )
        output_browse_btn.pack(side=tk.RIGHT)

        # 詳細オプションフレーム
        advanced_frame = ttk.LabelFrame(main_frame, text="詳細オプション", padding=10)
        advanced_frame.pack(fill=tk.X, pady=10)

        # AI分析インサイト生成
        self.use_ai_var = tk.BooleanVar(value=True)
        ai_check = ttk.Checkbutton(
            advanced_frame,
            text="AI分析インサイトを生成（OpenAI APIキーが必要）",
            variable=self.use_ai_var,
        )
        ai_check.pack(anchor=tk.W, pady=5)

        # 実行ボタン
        run_frame = ttk.Frame(main_frame)
        run_frame.pack(fill=tk.X, pady=20)

        self.run_btn = ttk.Button(
            run_frame, text="分析開始", command=self.start_analysis
        )
        self.run_btn.pack(side=tk.LEFT, padx=10)

        self.cancel_btn = ttk.Button(
            run_frame,
            text="キャンセル",
            command=self.cancel_analysis,
            state=tk.DISABLED,
        )
        self.cancel_btn.pack(side=tk.LEFT, padx=10)

        view_dir_btn = ttk.Button(
            run_frame, text="出力フォルダを開く", command=self.open_output_dir
        )
        view_dir_btn.pack(side=tk.RIGHT, padx=10)

        # ログ表示エリア
        log_frame = ttk.LabelFrame(main_frame, text="ログ", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # スクロールバー付きテキストウィジェット
        self.log_text = tk.Text(log_frame, height=10, wrap=tk.WORD)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)

        # ステータスバー
        self.status_var = tk.StringVar()
        self.status_var.set("準備完了")
        status_bar = ttk.Label(
            self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # 初期ログメッセージ
        self.log("自動データ分析ツールを起動しました")
        self.log(f"分析結果は以下のディレクトリに保存されます: {self.download_dir}")

    def browse_file(self):
        """ファイル選択ダイアログを表示"""
        filetypes = [
            ("データファイル", "*.csv;*.xlsx;*.xls;*.pdf"),
            ("CSVファイル", "*.csv"),
            ("Excelファイル", "*.xlsx;*.xls"),
            ("PDFファイル", "*.pdf"),
            ("すべてのファイル", "*.*"),
        ]

        file_path = filedialog.askopenfilename(
            title="分析するファイルを選択", filetypes=filetypes
        )

        if file_path:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, file_path)
            self.log(f"ファイル選択: {file_path}")

            # PDF選択時に警告表示
            if file_path.lower().endswith(".pdf"):
                has_pdf_libs = check_pdf_libraries()
                if not has_pdf_libs:
                    self.log(
                        "警告: PDFを処理するには追加ライブラリのインストールが必要です"
                    )
                    self.log("pip install tabula-py pdfplumber PyPDF2")

    def browse_output_dir(self):
        """出力ディレクトリ選択ダイアログを表示"""
        dir_path = filedialog.askdirectory(
            title="分析結果の出力先を選択", initialdir=self.download_dir
        )

        if dir_path:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, dir_path)
            self.log(f"出力先変更: {dir_path}")

    def start_analysis(self):
        """分析を開始する"""
        file_path = self.file_entry.get().strip()

        if not file_path:
            messagebox.showerror("エラー", "ファイルを選択してください")
            return

        if not os.path.exists(file_path):
            messagebox.showerror("エラー", "指定されたファイルが見つかりません")
            return

        # 分析設定の取得
        target_col = self.target_entry.get().strip() or None
        output_dir = self.output_entry.get().strip() or self.download_dir
        api_key = self.api_entry.get().strip() or None

        # 日付列の処理
        date_cols_str = self.date_entry.get().strip()
        date_cols = (
            [col.strip() for col in date_cols_str.split(",")] if date_cols_str else None
        )

        # 除外列の処理
        drop_cols_str = self.drop_entry.get().strip()
        drop_cols = (
            [col.strip() for col in drop_cols_str.split(",")] if drop_cols_str else None
        )

        # AI機能を使わない場合はAPIキーをクリア
        if not self.use_ai_var.get():
            api_key = None

        # 分析実行前の準備
        self.status_var.set("分析実行中...")
        self.run_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.NORMAL)

        # 分析を別スレッドで実行
        self.current_task = threading.Thread(
            target=self._run_analysis_thread,
            args=(file_path, target_col, output_dir, api_key, drop_cols, date_cols),
        )
        self.current_task.daemon = True
        self.current_task.start()

    def _run_analysis_thread(
        self, file_path, target_col, output_dir, api_key, drop_cols, date_cols
    ):
        """別スレッドで分析を実行する"""
        try:
            self.log(f"分析を開始します: {os.path.basename(file_path)}")
            self.log(f"出力先: {output_dir}")
            if target_col:
                self.log(f"目的変数: {target_col}")

            # sys.stdoutを一時的にリダイレクト
            original_stdout = sys.stdout
            sys.stdout = self

            # 分析実行
            self.analyzer = run_analysis(
                file_path=file_path,
                target_col=target_col,
                output_dir=output_dir,
                api_key=api_key,
                drop_cols=drop_cols,
                date_cols=date_cols,
            )

            # 標準出力を元に戻す
            sys.stdout = original_stdout

            if self.analyzer:
                self.log("分析が正常に完了しました")
                self.log(f"レポートファイル: {self.analyzer.report_file}")

                # GUIスレッドで処理を実行
                self.root.after(
                    0, lambda: self._analysis_completed(True, self.analyzer.report_file)
                )
            else:
                self.log("分析中にエラーが発生しました")

                # GUIスレッドで処理を実行
                self.root.after(0, lambda: self._analysis_completed(False))

        except Exception as e:
            self.log(f"エラーが発生しました: {str(e)}")

            # 標準出力を元に戻す
            sys.stdout = original_stdout

            # GUIスレッドで処理を実行
            self.root.after(0, lambda: self._analysis_completed(False))

    def _analysis_completed(self, success, report_file=None):
        """分析完了時の処理"""
        self.run_btn.config(state=tk.NORMAL)
        self.cancel_btn.config(state=tk.DISABLED)

        if success:
            self.status_var.set("分析完了")

            # 結果を表示するか尋ねる
            if messagebox.askyesno(
                "完了", "分析が完了しました。分析レポートを表示しますか？"
            ):
                self.open_report(report_file)
        else:
            self.status_var.set("分析失敗")
            messagebox.showerror(
                "エラー", "分析中にエラーが発生しました。ログを確認してください。"
            )

    def cancel_analysis(self):
        """実行中の分析をキャンセルする"""
        if messagebox.askyesno("確認", "実行中の分析をキャンセルしますか？"):
            self.log("分析をキャンセルしました")
            self.status_var.set("キャンセルされました")
            self.run_btn.config(state=tk.NORMAL)
            self.cancel_btn.config(state=tk.DISABLED)

            # 現在のスレッドは強制終了できないので、フラグで終了を通知
            # 実際のアプリケーションでは、もっと堅牢なキャンセル機構が必要
            if hasattr(self, "analyzer") and self.analyzer:
                self.analyzer._cancel_requested = True

    def open_output_dir(self):
        """出力ディレクトリをファイルエクスプローラで開く"""
        output_dir = self.output_entry.get().strip() or self.download_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        try:
            # プラットフォームに応じてディレクトリを開く
            if sys.platform == "win32":
                os.startfile(output_dir)
            elif sys.platform == "darwin":  # macOS
                subprocess.run(["open", output_dir])
            else:  # Linux
                subprocess.run(["xdg-open", output_dir])
        except Exception as e:
            messagebox.showerror("エラー", f"フォルダを開けませんでした: {str(e)}")

    def open_report(self, report_file):
        """分析レポートをブラウザで開く"""
        if report_file and os.path.exists(report_file):
            try:
                # ファイルのURLを作成
                file_url = f"file://{os.path.abspath(report_file)}"
                webbrowser.open(file_url)
            except Exception as e:
                messagebox.showerror("エラー", f"レポートを開けませんでした: {str(e)}")

    def log(self, message):
        """ログメッセージを追加"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)  # 最新の行までスクロール

        # GUIを更新
        self.root.update_idletasks()

    def write(self, message):
        """sys.stdoutリダイレクト用のメソッド"""
        self.log(message.rstrip())

    def flush(self):
        """sys.stdoutリダイレクト用のメソッド"""
        pass


def main():
    root = tk.Tk()
    app = AutoDataAnalysisGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
