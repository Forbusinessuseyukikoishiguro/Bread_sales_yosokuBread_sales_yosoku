# auto_data_analysis.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
import time
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# OpenAIが使えるか確認し、使えない場合はスキップする
try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI APIが利用できません。AIインサイト機能は無効になります。")


class AutoDataAnalysis:
    def __init__(self, api_key=None, output_dir=None):
        """
        データの自動分析と可視化を行うクラス

        Parameters:
        -----------
        api_key : str, optional
            OpenAI APIキー。環境変数 'OPENAI_API_KEY' からも取得可能
        output_dir : str, optional
            分析結果の出力ディレクトリ（Noneの場合はダウンロードフォルダを使用）
        """
        # 出力ディレクトリの設定
        if output_dir is None:
            # ダウンロードフォルダのパスを取得
            self.output_dir = os.path.join(
                os.path.expanduser("~"), "Downloads", "data_analysis_results"
            )
        else:
            self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)
        print(f"分析結果の出力先: {self.output_dir}")

        # 可視化ディレクトリの設定
        self.viz_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(self.viz_dir, exist_ok=True)

        # APIキーの設定（OpenAIが使える場合のみ）
        self.api_key = None
        if OPENAI_AVAILABLE:
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if self.api_key:
                openai.api_key = self.api_key

        # 分析結果の保存先
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_file = os.path.join(
            self.output_dir, f"analysis_report_{timestamp}.md"
        )

        # 結果の保存用辞書
        self.results = {
            "data_info": {},
            "preprocessing": {},
            "visualizations": [],
            "model_results": {},
            "insights": [],
        }

        # ログ開始
        with open(self.report_file, "w", encoding="utf-8") as f:
            f.write(f"# データ分析レポート\n\n")
            f.write(f"分析日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    def load_data(self, file_path, **kwargs):
        """
        データファイルを読み込む

        Parameters:
        -----------
        file_path : str
            読み込むファイルのパス
        **kwargs :
            pd.read_csvなどに渡す追加パラメータ

        Returns:
        --------
        pandas.DataFrame
            読み込んだデータフレーム
        """
        print(f"データを読み込んでいます: {file_path}")

        file_ext = file_path.split(".")[-1].lower()

        try:
            if file_ext == "csv":
                self.df = pd.read_csv(file_path, **kwargs)
            elif file_ext in ["xls", "xlsx"]:
                self.df = pd.read_excel(file_path, **kwargs)
            elif file_ext == "json":
                self.df = pd.read_json(file_path, **kwargs)
            elif file_ext == "parquet":
                self.df = pd.read_parquet(file_path, **kwargs)
            else:
                raise ValueError(f"未サポートのファイル形式です: {file_ext}")

            # データの基本情報を保存
            self.results["data_info"] = {
                "file_path": file_path,
                "shape": self.df.shape,
                "columns": list(self.df.columns),
                "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            }

            # レポートにデータ情報を追加
            with open(self.report_file, "a", encoding="utf-8") as f:
                f.write(f"## データ概要\n\n")
                f.write(f"- ファイル: {os.path.basename(file_path)}\n")
                f.write(f"- レコード数: {self.df.shape[0]}行\n")
                f.write(f"- 項目数: {self.df.shape[1]}列\n\n")

                f.write("### データ先頭部分\n\n")
                f.write("```\n")
                f.write(self.df.head().to_string())
                f.write("\n```\n\n")

                f.write("### データ型情報\n\n")
                f.write("```\n")
                f.write(self.df.dtypes.to_string())
                f.write("\n```\n\n")

            print(
                f"データを読み込みました: {self.df.shape[0]}行 x {self.df.shape[1]}列"
            )
            return self.df

        except Exception as e:
            print(f"データ読み込み中にエラーが発生しました: {e}")
            raise

    def analyze_data(self):
        """
        データの基本的な分析を実行

        Returns:
        --------
        dict
            分析結果を含む辞書
        """
        print("データの基本分析を実行しています...")

        # 基本統計量
        numeric_cols = self.df.select_dtypes(include=["int64", "float64"]).columns
        categorical_cols = self.df.select_dtypes(include=["object", "category"]).columns

        # 欠損値の分析
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        missing_info = pd.DataFrame(
            {"欠損値数": missing_data, "欠損率(%)": missing_percent}
        ).sort_values("欠損値数", ascending=False)

        # 相関分析
        if len(numeric_cols) > 1:
            corr_matrix = self.df[numeric_cols].corr()
            # 高い相関を持つ変数ペアを抽出
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.7:  # 相関係数の閾値
                        corr_pairs.append(
                            (
                                corr_matrix.columns[i],
                                corr_matrix.columns[j],
                                corr_matrix.iloc[i, j],
                            )
                        )

        # 分析結果をレポートに追加
        with open(self.report_file, "a", encoding="utf-8") as f:
            f.write(f"## データ分析\n\n")

            f.write("### 基本統計量\n\n")
            if len(numeric_cols) > 0:
                f.write("数値型データの統計:\n\n")
                f.write("```\n")
                f.write(self.df[numeric_cols].describe().to_string())
                f.write("\n```\n\n")

            if len(categorical_cols) > 0:
                f.write("カテゴリ型データの統計:\n\n")
                for col in categorical_cols:
                    f.write(f"**{col}** のカテゴリ値数: {self.df[col].nunique()}\n\n")
                    if self.df[col].nunique() < 10:  # カテゴリ数が少ない場合は詳細表示
                        f.write("```\n")
                        f.write(self.df[col].value_counts().to_string())
                        f.write("\n```\n\n")

            f.write("### 欠損値分析\n\n")
            f.write("```\n")
            f.write(missing_info[missing_info["欠損値数"] > 0].to_string())
            f.write("\n```\n\n")

            if len(numeric_cols) > 1:
                f.write("### 相関分析\n\n")
                if corr_pairs:
                    f.write("高い相関(>0.7)を持つ変数ペア:\n\n")
                    for col1, col2, corr in corr_pairs:
                        f.write(f"- {col1} と {col2}: {corr:.4f}\n")
                    f.write("\n")
                else:
                    f.write("高い相関を持つ変数ペアは見つかりませんでした\n\n")

        # LLMを使った分析インサイト生成
        if OPENAI_AVAILABLE and self.api_key:
            self._generate_insights()

        print("基本分析が完了しました")
        return self.results

    def predict(self, new_data, return_proba=False):
        """
        新しいデータに対して予測を行う

        Parameters:
        -----------
        new_data : pandas.DataFrame
            予測を行う新しいデータ
        return_proba : bool, default=False
            確率を返すかどうか（分類タスクのみ）

        Returns:
        --------
        numpy.ndarray
            予測結果
        """
        if not hasattr(self, "model") or not hasattr(self, "preprocessor"):
            print("予測を行うには先にモデルを訓練する必要があります。")
            return None

        print("新しいデータに対して予測を行っています...")

        # 前処理を適用
        X_new_processed = self.preprocessor.transform(new_data)

        # 予測
        if (
            return_proba
            and hasattr(self.model, "predict_proba")
            and self.task_type == "classification"
        ):
            predictions = self.model.predict_proba(X_new_processed)
            print(f"確率的予測が完了しました: {len(predictions)}件")
            return predictions
        else:
            predictions = self.model.predict(X_new_processed)

            # 分類タスクの場合、元のラベルに変換
            if self.task_type == "classification" and hasattr(self, "label_encoder"):
                predictions = self.label_encoder.inverse_transform(predictions)

            print(f"予測が完了しました: {len(predictions)}件")
            return predictions

    def save_model(self, model_path=None):
        """
        モデルを保存する

        Parameters:
        -----------
        model_path : str, optional
            モデルの保存先パス。Noneの場合はデフォルトパスを使用

        Returns:
        --------
        str
            保存先パス
        """
        if not hasattr(self, "model") or not hasattr(self, "preprocessor"):
            print("保存するモデルがありません。")
            return None

        if model_path is None:
            model_path = os.path.join(self.output_dir, "model.pkl")

        import pickle

        # モデル情報を辞書にまとめる
        model_info = {
            "model": self.model,
            "preprocessor": self.preprocessor,
            "task_type": self.task_type,
            "target_col": self.target_col,
            "model_name": self.model_name,
        }

        # ラベルエンコーダーがある場合は追加
        if hasattr(self, "label_encoder"):
            model_info["label_encoder"] = self.label_encoder

        # モデルを保存
        try:
            with open(model_path, "wb") as f:
                pickle.dump(model_info, f)
            print(f"モデルを保存しました: {model_path}")
            return model_path
        except Exception as e:
            print(f"モデル保存中にエラーが発生しました: {e}")
            return None

    @classmethod
    def load_model(cls, model_path):
        """
        保存されたモデルを読み込む

        Parameters:
        -----------
        model_path : str
            モデルファイルのパス

        Returns:
        --------
        AutoDataAnalysis
            モデルを読み込んだインスタンス
        """
        import pickle

        try:
            with open(model_path, "rb") as f:
                model_info = pickle.load(f)

            # 新しいインスタンスを作成
            instance = cls()

            # モデル情報を設定
            for key, value in model_info.items():
                setattr(instance, key, value)

            print(f"モデルを読み込みました: {model_path}")
            return instance
        except Exception as e:
            print(f"モデル読み込み中にエラーが発生しました: {e}")
            return None

    def run_pipeline(
        self,
        file_path,
        target_col=None,
        categorical_cols=None,
        numeric_cols=None,
        drop_cols=None,
        date_cols=None,
        model=None,
        test_size=0.2,
    ):
        """
        データ分析と予測のパイプラインを一気通貫で実行する

        Parameters:
        -----------
        file_path : str
            データファイルのパス
        target_col : str, optional
            目的変数の列名
        categorical_cols : list, optional
            カテゴリ変数の列名リスト
        numeric_cols : list, optional
            数値変数の列名リスト
        drop_cols : list, optional
            除外する列名リスト
        date_cols : list, optional
            日付変数の列名リスト
        model : estimator, optional
            使用するモデル
        test_size : float, default=0.2
            テストデータの割合

        Returns:
        --------
        dict
            実行結果を含む辞書
        """
        print("データ分析と予測のパイプラインを実行します...")

        # 1. データ読み込み
        self.load_data(file_path)

        # 2. データ分析
        self.analyze_data()

        # 3. データ前処理
        self.preprocess_data(
            target_col=target_col,
            categorical_cols=categorical_cols,
            numeric_cols=numeric_cols,
            drop_cols=drop_cols,
            date_cols=date_cols,
            test_size=test_size,
        )

        # 4. データ可視化
        self.visualize_data()

        # 5. モデル訓練（目的変数がある場合のみ）
        if target_col:
            self.train_model(model=model)

            # モデルを保存
            self.save_model()

        print("パイプラインの実行が完了しました")
        print(f"レポートファイル: {self.report_file}")

        return self.results

    def _generate_insights(self):
        """ChatGPT APIを使ってデータインサイトを生成"""
        try:
            # データ概要と統計情報を文字列にまとめる
            data_description = f"""
データセット情報:
- レコード数: {self.df.shape[0]}行
- 項目数: {self.df.shape[1]}列
- カラム: {', '.join(self.df.columns.tolist())}

基本統計量:
{self.df.describe().to_string()}

カテゴリ項目の値:
{str({col: self.df[col].value_counts().to_dict() for col in self.df.select_dtypes(include=['object', 'category']).columns if self.df[col].nunique() < 10})}

欠損値情報:
{self.df.isnull().sum().to_string()}
            """

            response = openai.ChatCompletion.create(
                model="gpt-4",  # または "gpt-3.5-turbo"
                messages=[
                    {
                        "role": "system",
                        "content": "あなたはデータサイエンティストです。データに基づいた洞察を提供してください。",
                    },
                    {
                        "role": "user",
                        "content": f"以下のデータセットを分析し、重要なパターン、異常、および次に探求すべき方向性を3〜5つ箇条書きで簡潔に示してください。専門家として、データから得られるビジネスインサイトも提案してください。\n\n{data_description}",
                    },
                ],
            )

            insights = response.choices[0].message.content
            self.results["insights"].append(insights)

            # レポートに追加
            with open(self.report_file, "a", encoding="utf-8") as f:
                f.write("### AIによるデータインサイト\n\n")
                f.write(insights)
                f.write("\n\n")

            print("AIによるデータインサイトを生成しました")
        except Exception as e:
            print(f"AIインサイト生成中にエラーが発生しました: {e}")

    def preprocess_data(
        self,
        target_col=None,
        categorical_cols=None,
        numeric_cols=None,
        drop_cols=None,
        date_cols=None,
        test_size=0.2,
    ):
        """
        データの前処理を実行

        Parameters:
        -----------
        target_col : str, optional
            目的変数の列名
        categorical_cols : list, optional
            カテゴリ変数の列名リスト (Noneの場合は自動検出)
        numeric_cols : list, optional
            数値変数の列名リスト (Noneの場合は自動検出)
        drop_cols : list, optional
            除外する列名リスト
        date_cols : list, optional
            日付変数の列名リスト
        test_size : float, default=0.2
            テストデータの割合

        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test) の組、またはtarget_colがNoneの場合はX_processed
        """
        print("データの前処理を実行しています...")

        df_proc = self.df.copy()

        # 列の除外
        if drop_cols:
            df_proc = df_proc.drop(
                columns=[col for col in drop_cols if col in df_proc.columns]
            )
            print(f"除外した列: {drop_cols}")

        # 日付列の処理
        if date_cols:
            for col in date_cols:
                if col in df_proc.columns:
                    try:
                        df_proc[col] = pd.to_datetime(df_proc[col])
                        # 年月日の特徴量を追加
                        df_proc[f"{col}_year"] = df_proc[col].dt.year
                        df_proc[f"{col}_month"] = df_proc[col].dt.month
                        df_proc[f"{col}_day"] = df_proc[col].dt.day
                        df_proc[f"{col}_dayofweek"] = df_proc[col].dt.dayofweek

                        # 元の日付列を削除
                        df_proc = df_proc.drop(columns=[col])

                        print(f"日付列 {col} を特徴量に変換しました")
                    except Exception as e:
                        print(f"日付列 {col} の変換中にエラーが発生しました: {e}")

        # カテゴリ列と数値列の自動検出
        if categorical_cols is None:
            categorical_cols = df_proc.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
            print(f"検出されたカテゴリ列: {categorical_cols}")

        if numeric_cols is None:
            numeric_cols = df_proc.select_dtypes(
                include=["int64", "float64"]
            ).columns.tolist()
            print(f"検出された数値列: {numeric_cols}")

        # 目的変数と説明変数の分離
        if target_col:
            if target_col not in df_proc.columns:
                raise ValueError(f"目的変数 {target_col} がデータに存在しません")

            if target_col in numeric_cols:
                numeric_cols.remove(target_col)
            if target_col in categorical_cols:
                categorical_cols.remove(target_col)

            y = df_proc[target_col]
            X = df_proc.drop(columns=[target_col])

            task_type = (
                "regression" if y.dtype in ["int64", "float64"] else "classification"
            )
            print(f"検出されたタスク: {task_type}")

            # 分類タスクの場合はターゲットをエンコード
            if task_type == "classification":
                le = LabelEncoder()
                y = le.fit_transform(y)
                self.label_encoder = le
                self.classes_ = le.classes_
                print(f"分類クラス: {le.classes_}")
        else:
            X = df_proc
            y = None
            task_type = None

        # 前処理パイプラインの構築
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, categorical_cols),
            ]
        )

        # 前処理の実行
        if y is not None:
            # 訓練データとテストデータに分割
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            # 前処理パイプラインを適用
            self.preprocessor = preprocessor
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)

            # 前処理結果をレポートに追加
            with open(self.report_file, "a", encoding="utf-8") as f:
                f.write(f"## データ前処理\n\n")
                f.write(f"- 訓練データ: {X_train.shape[0]}行\n")
                f.write(f"- テストデータ: {X_test.shape[0]}行\n")
                f.write(f"- 処理されたカテゴリ列: {categorical_cols}\n")
                f.write(f"- 処理された数値列: {numeric_cols}\n")
                if date_cols:
                    f.write(f"- 処理された日付列: {date_cols}\n")
                f.write(f"- タスクタイプ: {task_type}\n\n")

            # 前処理結果を保存
            self.results["preprocessing"] = {
                "train_size": X_train.shape[0],
                "test_size": X_test.shape[0],
                "categorical_cols": categorical_cols,
                "numeric_cols": numeric_cols,
                "date_cols": date_cols,
                "task_type": task_type,
            }

            # モデル学習に必要な情報を保存
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
            self.X_train_processed = X_train_processed
            self.X_test_processed = X_test_processed
            self.task_type = task_type
            self.target_col = target_col

            print(
                f"前処理が完了しました: {X_train_processed.shape[1]}個の特徴量が生成されました"
            )
            return X_train, X_test, y_train, y_test
        else:
            # 前処理パイプラインを適用（目的変数なしの場合）
            self.preprocessor = preprocessor
            X_processed = preprocessor.fit_transform(X)

            # 前処理結果をレポートに追加
            with open(self.report_file, "a", encoding="utf-8") as f:
                f.write(f"## データ前処理\n\n")
                f.write(f"- 処理されたカテゴリ列: {categorical_cols}\n")
                f.write(f"- 処理された数値列: {numeric_cols}\n")
                if date_cols:
                    f.write(f"- 処理された日付列: {date_cols}\n\n")

            print(
                f"前処理が完了しました: {X_processed.shape[1]}個の特徴量が生成されました"
            )
            return X_processed

    def visualize_data(self, save_figs=True):
        """
        データを可視化する

        Parameters:
        -----------
        save_figs : bool, default=True
            図を保存するかどうか

        Returns:
        --------
        list
            生成された図のリスト
        """
        print("データの可視化を実行しています...")

        figs = []
        viz_paths = []

        # 数値列のヒストグラム
        numeric_cols = self.df.select_dtypes(include=["int64", "float64"]).columns
        if len(numeric_cols) > 0:
            # 1行に最大3つのヒストグラムを表示
            n_cols = min(3, len(numeric_cols))
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
            axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

            for i, col in enumerate(numeric_cols):
                axes[i].hist(self.df[col].dropna(), bins=30, alpha=0.7)
                axes[i].set_title(f"{col} の分布")
                axes[i].set_xlabel(col)
                axes[i].set_ylabel("頻度")

            # 使用していない軸を非表示
            for j in range(i + 1, len(axes)):
                axes[j].axis("off")

            plt.tight_layout()

            if save_figs:
                plt_path = os.path.join(self.viz_dir, "numeric_histograms.png")
                plt.savefig(plt_path)
                viz_paths.append(plt_path)

            figs.append(fig)

            # レポートに追加
            with open(self.report_file, "a", encoding="utf-8") as f:
                f.write(f"## データ可視化\n\n")
                f.write(f"### 数値変数の分布\n\n")
                f.write(
                    f"![数値変数のヒストグラム](visualizations/numeric_histograms.png)\n\n"
                )

        # カテゴリ列の棒グラフ
        categorical_cols = self.df.select_dtypes(include=["object", "category"]).columns
        for col in categorical_cols:
            # カテゴリが10個以下の場合のみ表示
            if self.df[col].nunique() <= 10:
                plt.figure(figsize=(10, 6))
                self.df[col].value_counts().plot(kind="bar")
                plt.title(f"{col} のカテゴリ分布")
                plt.xlabel(col)
                plt.ylabel("頻度")
                plt.xticks(rotation=45)
                plt.tight_layout()

                if save_figs:
                    plt_path = os.path.join(self.viz_dir, f"category_{col}.png")
                    plt.savefig(plt_path)
                    viz_paths.append(plt_path)

                figs.append(plt.gcf())

                # レポートに追加
                with open(self.report_file, "a", encoding="utf-8") as f:
                    f.write(f"### {col} のカテゴリ分布\n\n")
                    f.write(f"![{col} の分布](visualizations/category_{col}.png)\n\n")

        # 数値変数間の相関行列ヒートマップ
        if len(numeric_cols) > 1:
            plt.figure(figsize=(max(10, len(numeric_cols)), max(8, len(numeric_cols))))
            corr_matrix = self.df[numeric_cols].corr()
            sns.heatmap(
                corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5
            )
            plt.title("相関行列")
            plt.tight_layout()

            if save_figs:
                plt_path = os.path.join(self.viz_dir, "correlation_matrix.png")
                plt.savefig(plt_path)
                viz_paths.append(plt_path)

            figs.append(plt.gcf())

            # レポートに追加
            with open(self.report_file, "a", encoding="utf-8") as f:
                f.write(f"### 相関行列\n\n")
                f.write(f"![相関行列](visualizations/correlation_matrix.png)\n\n")

        # 目的変数が存在する場合の可視化
        if hasattr(self, "task_type") and self.task_type:
            target_col = self.target_col

            # 目的変数の分布
            plt.figure(figsize=(10, 6))
            if self.task_type == "regression":
                plt.hist(self.df[target_col].dropna(), bins=30, alpha=0.7)
                plt.title(f"目的変数 {target_col} の分布")
            else:  # classification
                self.df[target_col].value_counts().plot(kind="bar")
                plt.title(f"目的変数 {target_col} のクラス分布")

            plt.xlabel(target_col)
            plt.ylabel("頻度")
            plt.tight_layout()

            if save_figs:
                plt_path = os.path.join(self.viz_dir, "target_distribution.png")
                plt.savefig(plt_path)
                viz_paths.append(plt_path)

            figs.append(plt.gcf())

            # レポートに追加
            with open(self.report_file, "a", encoding="utf-8") as f:
                f.write(f"### 目的変数 {target_col} の分布\n\n")
                f.write(
                    f"![目的変数の分布](visualizations/target_distribution.png)\n\n"
                )

            # 主要な特徴量と目的変数の関係
            if self.task_type == "regression" and len(numeric_cols) > 0:
                # 数値特徴量と目的変数の散布図（上位5つ）
                corr_matrix = self.df[list(numeric_cols) + [target_col]].corr()
                top_features = (
                    corr_matrix[target_col]
                    .abs()
                    .sort_values(ascending=False)
                    .index[1:6]
                )

                n_features = len(top_features)
                if n_features > 0:
                    n_cols = min(2, n_features)
                    n_rows = (n_features + n_cols - 1) // n_cols

                    fig, axes = plt.subplots(
                        n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5)
                    )
                    if n_rows * n_cols > 1:
                        axes = axes.flatten()
                    else:
                        axes = [axes]

                    for i, feature in enumerate(top_features):
                        sns.regplot(x=feature, y=target_col, data=self.df, ax=axes[i])
                        axes[i].set_title(f"{feature} vs {target_col}")

                    # 使用していない軸を非表示
                    for j in range(i + 1, len(axes)):
                        axes[j].axis("off")

                    plt.tight_layout()

                    if save_figs:
                        plt_path = os.path.join(
                            self.viz_dir, "feature_target_relationships.png"
                        )
                        plt.savefig(plt_path)
                        viz_paths.append(plt_path)

                    figs.append(fig)

                    # レポートに追加
                    with open(self.report_file, "a", encoding="utf-8") as f:
                        f.write(f"### 主要な特徴量と目的変数の関係\n\n")
                        f.write(
                            f"![特徴量と目的変数の関係](visualizations/feature_target_relationships.png)\n\n"
                        )

        # 可視化結果を保存
        self.results["visualizations"] = viz_paths

        print(f"可視化が完了しました: {len(figs)}個のグラフを生成しました")
        return figs

    def train_model(self, model=None):
        """
        機械学習モデルを訓練する

        Parameters:
        -----------
        model : estimator, optional
            学習に使用するモデル (Noneの場合はタスクタイプに応じて自動選択)

        Returns:
        --------
        estimator
            訓練されたモデル
        """
        if not hasattr(self, "task_type") or not self.task_type:
            print(
                "モデル訓練には目的変数の指定が必要です。preprocess_data()を先に実行してください。"
            )
            return None

        print(f"{self.task_type}モデルの訓練を実行しています...")

        # 元の目的変数名を使用
        target_col = self.target_col

        # モデルの選択
        if model is None:
            if self.task_type == "regression":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                print("ランダムフォレスト回帰モデルを使用します")
            else:  # classification
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                print("ランダムフォレスト分類モデルを使用します")

        # モデルの識別情報を保存
        self.model_name = model.__class__.__name__

        # モデルの訓練
        model.fit(self.X_train_processed, self.y_train)
        self.model = model

        # 予測と評価
        y_pred = model.predict(self.X_test_processed)

        # ChatGPT APIを使って予測結果を解釈
        if OPENAI_AVAILABLE and self.api_key and len(y_pred) > 0:
            self._interpret_predictions(y_pred)

        # 評価指標の計算
        if self.task_type == "regression":
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)

            # モデル結果を保存
            self.results["model_results"] = {
                "model_type": model.__class__.__name__,
                "metrics": {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2},
            }

            eval_metrics = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R²": r2}

            # 予測と実測値の散布図
            plt.figure(figsize=(10, 6))
            plt.scatter(self.y_test, y_pred, alpha=0.5)
            plt.plot(
                [self.y_test.min(), self.y_test.max()],
                [self.y_test.min(), self.y_test.max()],
                "k--",
                lw=2,
            )
            plt.xlabel("実測値")
            plt.ylabel("予測値")
            plt.title("予測値 vs 実測値")

            plt_path = os.path.join(self.output_dir, "pred_vs_actual.png")
            plt.savefig(plt_path)

            # 残差プロット
            plt.figure(figsize=(10, 6))
            residuals = self.y_test - y_pred
            plt.scatter(y_pred, residuals, alpha=0.5)
            plt.axhline(y=0, color="k", linestyle="--", lw=2)
            plt.xlabel("予測値")
            plt.ylabel("残差")
            plt.title("残差プロット")

            residual_plt_path = os.path.join(self.output_dir, "residual_plot.png")
            plt.savefig(residual_plt_path)

            # レポートに追加
            with open(self.report_file, "a", encoding="utf-8") as f:
                f.write(f"## モデル学習結果\n\n")
                f.write(f"モデル: {model.__class__.__name__}\n\n")
                f.write(f"### 評価指標\n\n")
                f.write(f"- 平均二乗誤差 (MSE): {mse:.4f}\n")
                f.write(f"- 平方根平均二乗誤差 (RMSE): {rmse:.4f}\n")
                f.write(f"- 平均絶対誤差 (MAE): {mae:.4f}\n")
                f.write(f"- 決定係数 (R²): {r2:.4f}\n\n")
                f.write(f"### 予測値 vs 実測値\n\n")
                f.write(f"![予測値 vs 実測値](pred_vs_actual.png)\n\n")
                f.write(f"### 残差プロット\n\n")
                f.write(f"![残差プロット](residual_plot.png)\n\n")

        else:  # classification
            accuracy = accuracy_score(self.y_test, y_pred)
            report = classification_report(self.y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(self.y_test, y_pred)

            # モデル結果を保存
            self.results["model_results"] = {
                "model_type": model.__class__.__name__,
                "metrics": {"accuracy": accuracy, "classification_report": report},
            }

            # 混同行列のプロット
            plt.figure(figsize=(10, 8))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
            plt.xlabel("予測クラス")
            plt.ylabel("実際のクラス")
            plt.title("混同行列")

            conf_matrix_path = os.path.join(self.output_dir, "confusion_matrix.png")
            plt.savefig(conf_matrix_path)

            # レポートに追加
            with open(self.report_file, "a", encoding="utf-8") as f:
                f.write(f"## モデル学習結果\n\n")
                f.write(f"モデル: {model.__class__.__name__}\n\n")
                f.write(f"### 評価指標\n\n")
                f.write(f"- 精度 (Accuracy): {accuracy:.4f}\n\n")
                f.write(f"### 分類レポート\n\n")
                f.write("```\n")
                f.write(classification_report(self.y_test, y_pred))
                f.write("\n```\n\n")
                f.write(f"### 混同行列\n\n")
                f.write(f"![混同行列](confusion_matrix.png)\n\n")

        # 特徴量重要度
        if hasattr(model, "feature_importances_"):
            feature_names = []

            # 数値特徴量の名前を取得
            if hasattr(self.preprocessor, "transformers_"):
                for name, transformer, cols in self.preprocessor.transformers_:
                    if name == "num":
                        feature_names.extend(cols)
                    elif name == "cat":
                        # OneHotEncoderの場合、カテゴリ名を取得
                        if (
                            hasattr(transformer, "named_steps")
                            and "encoder" in transformer.named_steps
                        ):
                            encoder = transformer.named_steps["encoder"]
                            if hasattr(encoder, "categories_"):
                                for i, category in enumerate(encoder.categories_):
                                    for cat in category:
                                        feature_names.append(f"{cols[i]}_{cat}")

            # 特徴量数と重要度の長さが一致しない場合は簡易的なインデックスを使用
            if len(feature_names) != len(model.feature_importances_):
                feature_names = [
                    f"Feature_{i}" for i in range(len(model.feature_importances_))
                ]

            # 特徴量重要度をプロット
            importances = pd.DataFrame(
                {"Feature": feature_names, "Importance": model.feature_importances_}
            ).sort_values("Importance", ascending=False)

            plt.figure(figsize=(12, 8))
            sns.barplot(x="Importance", y="Feature", data=importances.head(20))
            plt.title("特徴量重要度（上位20）")
            plt.tight_layout()

            importance_plt_path = os.path.join(
                self.output_dir, "feature_importance.png"
            )
            plt.savefig(importance_plt_path)

            # レポートに追加
            with open(self.report_file, "a", encoding="utf-8") as f:
                f.write(f"### 特徴量重要度\n\n")
                f.write(f"![特徴量重要度](feature_importance.png)\n\n")
                f.write("特徴量重要度（上位10）:\n\n")
                f.write("```\n")
                f.write(importances.head(10).to_string(index=False))
                f.write("\n```\n\n")

        print(f"モデル訓練が完了しました")
        return model

    def _interpret_predictions(self, y_pred):
        """ChatGPT APIを使って予測結果を解釈"""
        try:
            # 予測結果のサンプルを取得
            sample_size = min(10, len(y_pred))
            sample_indices = np.random.choice(len(y_pred), sample_size, replace=False)

            if self.task_type == "regression":
                sample_data = pd.DataFrame(
                    {
                        "実測値": self.y_test.iloc[sample_indices],
                        "予測値": y_pred[sample_indices],
                        "誤差": self.y_test.iloc[sample_indices]
                        - y_pred[sample_indices],
                    }
                )

                metrics = {
                    "mse": mean_squared_error(self.y_test, y_pred),
                    "rmse": np.sqrt(mean_squared_error(self.y_test, y_pred)),
                    "mae": mean_absolute_error(self.y_test, y_pred),
                    "r2": r2_score(self.y_test, y_pred),
                }

                prediction_info = f"""
    モデル: {self.model_name}
    目的変数: {self.target_col}
    サンプルデータ:
    {sample_data.to_string()}

    評価指標:
    - MSE: {metrics['mse']:.4f}
    - RMSE: {metrics['rmse']:.4f}
    - MAE: {metrics['mae']:.4f}
    - R²: {metrics['r2']:.4f}
                """
            else:  # classification
                if hasattr(self, "classes_"):
                    # 元のクラスラベルに変換
                    y_test_labels = [
                        self.classes_[i] for i in self.y_test.iloc[sample_indices]
                    ]
                    y_pred_labels = [self.classes_[i] for i in y_pred[sample_indices]]

                    sample_data = pd.DataFrame(
                        {
                            "実際のクラス": y_test_labels,
                            "予測クラス": y_pred_labels,
                            "一致": [
                                a == b for a, b in zip(y_test_labels, y_pred_labels)
                            ],
                        }
                    )
                else:
                    sample_data = pd.DataFrame(
                        {
                            "実際のクラス": self.y_test.iloc[sample_indices],
                            "予測クラス": y_pred[sample_indices],
                            "一致": self.y_test.iloc[sample_indices]
                            == y_pred[sample_indices],
                        }
                    )

                accuracy = accuracy_score(self.y_test, y_pred)
                report = classification_report(self.y_test, y_pred)

                prediction_info = f"""
    モデル: {self.model_name}
    目的変数: {self.target_col}
    サンプルデータ:
    {sample_data.to_string()}

    評価指標:
    - 精度: {accuracy:.4f}

    分類レポート:
    {report}
                """

            # AI インサイト生成処理（OpenAI API のバージョン問題があるため、この部分はコメントアウトまたは修正）
            # OpenAI API のバージョンが新しい場合の対応コード
            try:
                if OPENAI_AVAILABLE and hasattr(openai, "ChatCompletion"):
                    # 古いバージョンの API 呼び出し (openai < 1.0.0)
                    response = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=[
                            {
                                "role": "system",
                                "content": "あなたはデータサイエンティストです。機械学習モデルの予測結果を分析し、専門家としての洞察を提供してください。",
                            },
                            {
                                "role": "user",
                                "content": f"以下は機械学習モデルの予測結果です。この結果を分析し、モデルの性能、傾向、改善点、および実際のビジネス的インサイトを3〜5つ箇条書きで簡潔に説明してください。\n\n{prediction_info}",
                            },
                        ],
                    )
                    interpretation = response.choices[0].message.content

                    # レポートに追加
                    with open(self.report_file, "a", encoding="utf-8") as f:
                        f.write("### AIによる予測結果解釈\n\n")
                        f.write(interpretation)
                        f.write("\n\n")

                    print("AIによる予測結果の解釈を生成しました")
                else:
                    print("予測結果の解釈をスキップします (OpenAI API 利用不可)")
            except Exception as e:
                print(f"OpenAI API 呼び出しに失敗しました: {e}")
                print("予測結果の解釈をスキップします")

        except Exception as e:
            print(f"AI解釈生成中にエラーが発生しました: {e}")
