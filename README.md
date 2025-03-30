# 自動データ分析ツール

CSVやPDFファイルを読み込んで、前処理、分析、可視化、予測を自動的に行うPythonツールです。

## 特徴

- **CSVとPDFに対応**: CSVファイルだけでなく、PDFからもテーブルデータを抽出して分析
- **前処理の自動化**: カテゴリ変数のエンコード、数値変数の標準化、日付変数の特徴量エンジニアリング
- **可視化の自動生成**: ヒストグラム、散布図、相関ヒートマップなど
- **自動モデル構築**: 回帰・分類タスクに応じたモデル選択と学習
- **AI分析インサイト**: OpenAI APIを活用した高度なデータ解釈（オプション）
- **ダウンロードフォルダに保存**: 分析結果がダウンロードフォルダに自動保存
- **GUIインターフェース**: 使いやすいグラフィカルインターフェース

## インストール方法

### 1. 必要なライブラリのインストール

```bash
# 仮想環境を作成（venvの場合）
python -m venv data_analysis_env

# 仮想環境を有効化
# Windows の場合:
data_analysis_env\Scripts\activate

# macOS/Linux の場合:
source data_analysis_env/bin/activate

# 基本ライブラリをインストール
pip install pandas numpy matplotlib seaborn scikit-learn

# PDF処理用ライブラリをインストール（オプション）
pip install tabula-py pdfplumber PyPDF2

# OpenAI API用ライブラリをインストール（オプション）
pip install openai
```

### 2. ファイル構成

以下の3つのファイルを同じディレクトリに配置します：

1. `auto_data_analysis.py` - メインの分析エンジン
2. `pdf_extractor.py` - PDFからのデータ抽出モジュール
3. `run_analysis.py` - コマンドライン実行スクリプト
4. `gui_app.py` - GUIアプリケーション（オプション）

## 実行方法

### 方法1: GUIアプリケーションを使用（推奨）

```bash
# 仮想環境を有効化した状態で
python gui_app.py
```

GUI画面で以下の手順を実行します：

1. 「参照」ボタンをクリックしてCSVまたはPDFファイルを選択
2. 分析設定を入力（目的変数、除外列など）
3. 「分析開始」ボタンをクリック
4. 分析完了後、「出力フォルダを開く」でダウンロードフォルダを表示

### 方法2: コマンドラインから実行

```bash
# 仮想環境を有効化した状態で
python run_analysis.py データファイル.csv --target 目的変数 --output 出力ディレクトリ
```

例:
```bash
# CSVファイルを分析（目的変数: 価格）
python run_analysis.py house_data.csv --target price

# PDFファイルからデータを抽出して分析
python run_analysis.py report.pdf --target revenue --date order_date

# 日付列と除外列を指定して分析
python run_analysis.py sales.csv --target sales --date order_date --drop id customer_id
```

### 方法3: Pythonコードから使用

```python
from auto_data_analysis import AutoDataAnalysis
from run_analysis import run_analysis

# 方法A: 直接関数を使用
analyzer = run_analysis(
    file_path="your_data.csv",
    target_col="price",
    output_dir="~/Downloads/analysis_results"
)

# 方法B: クラスを詳細に使用
analyzer = AutoDataAnalysis(output_dir="~/Downloads/analysis_results")
analyzer.load_data("your_data.csv")
analyzer.analyze_data()
analyzer.preprocess_data(target_col="price", date_cols=["date"])
analyzer.visualize_data()
analyzer.train_model()
```

## 出力ファイル

分析実行後、以下のファイルが作成されます：

1. `analysis_report_[日時].md` - 分析レポート（Markdown形式）
2. `visualizations/` - グラフ画像のディレクトリ
3. `pred_vs_actual.png` - 予測と実測値の比較グラフ（回帰タスクの場合）
4. `confusion_matrix.png` - 混同行列（分類タスクの場合）
5. `feature_importance.png` - 特徴量重要度
6. `model.pkl` - 保存されたモデル

## トラブルシューティング

### PDFファイルからテーブルを抽出できない場合

PDFファイルの構造によっては、テーブルを正確に抽出できない場合があります。その場合は以下をお試しください：

1. PDFからテーブルをコピーして手動でCSVに貼り付ける
2. Adobe AcrobatやMicrosoft ExcelでPDFを開き、CSVとして保存する

### OpenAI APIのエラー

AIインサイト機能を使用するには、有効なOpenAI APIキーが必要です：

1. OpenAIのウェブサイトでAPIキーを取得
2. APIキーを環境変数として設定するか、GUIアプリケーションで入力
3. APIキーが無効な場合は、AI機能をオフにして基本分析のみを実行

### メモリエラー

大規模なデータセットを分析する場合、メモリ不足エラーが発生する可能性があります：

1. データサイズを縮小（行や列を減らす）
2. 仮想環境に割り当てるメモリを増やす
3. より高性能なマシンで実行する

## 注意事項

1. **PDFデータ抽出の制限**: 複雑なレイアウトのPDFからは正確にデータを抽出できない場合があります
2. **APIキーの管理**: OpenAI APIキーは安全に管理し、公開リポジトリにコミットしないでください
3. **分析時間**: 大きなデータセットや複雑な分析では、処理に時間がかかることがあります

## カスタマイズ

より高度な分析を行うには、以下のようにカスタマイズできます：

### 異なるモデルを使用

```python
from sklearn.ensemble import GradientBoostingRegressor
from auto_data_analysis import AutoDataAnalysis

analyzer = AutoDataAnalysis()
analyzer.load_data("your_data.csv")
analyzer.preprocess_data(target_col="price")

# カスタムモデルを使用
custom_model = GradientBoostingRegressor(n_estimators=200)
analyzer.train_model(model=custom_model)
```

### 追加の可視化を作成

```python
import matplotlib.pyplot as plt
import seaborn as sns
from auto_data_analysis import AutoDataAnalysis

analyzer = AutoDataAnalysis()
df = analyzer.load_data("your_data.csv")

# カスタム可視化
plt.figure(figsize=(10, 6))
sns.pairplot(df[['col1', 'col2', 'col3']])
plt.savefig("custom_viz.png")
```

## ライセンス

MITライセンス

## 貢献

貢献やフィードバックを歓迎します。バグレポートや機能リクエストは、Issueトラッカーに投稿してください。

----------

# 自動データ分析ツール - インストールと実行ガイド

このガイドでは、自動データ分析ツールのインストール方法と実行方法を詳しく説明します。

## 前提条件

- Python 3.7以上
- 基本的なコマンドラインの知識

## インストール手順

### ステップ1: 仮想環境の作成と有効化

まず、Pythonの仮想環境を作成して、他の環境と分離します。

**Windowsの場合:**

```bash
# 仮想環境を作成
python -m venv data_analysis_env

# 仮想環境を有効化
data_analysis_env\Scripts\activate
```

**macOS/Linuxの場合:**

```bash
# 仮想環境を作成
python -m venv data_analysis_env

# 仮想環境を有効化
source data_analysis_env/bin/activate
```

### ステップ2: 必要なライブラリのインストール

仮想環境を有効化した状態で、以下のコマンドを実行してライブラリをインストールします。

```bash
# 基本的な分析ライブラリ（必須）
pip install pandas numpy matplotlib seaborn scikit-learn

# PDFデータ抽出用のライブラリ（PDFを分析する場合に必要）
pip install tabula-py pdfplumber PyPDF2

# AIインサイト機能用のライブラリ（任意）
pip install openai
```

### ステップ3: ソースコードのダウンロードと配置

1. このリポジトリから4つのPythonファイルをダウンロードし、同じディレクトリに配置します：
   - `auto_data_analysis.py`（メイン分析エンジン）
   - `pdf_extractor.py`（PDFデータ抽出モジュール）
   - `run_analysis.py`（コマンドライン実行スクリプト）
   - `gui_app.py`（グラフィカルインターフェース）

2. ファイルの配置例:
   ```
   my_data_analysis/
   ├── auto_data_analysis.py
   ├── pdf_extractor.py
   ├── run_analysis.py
   └── gui_app.py
   ```

## 実行方法

### 方法1: GUIアプリケーションで実行（初心者向け）

GUIアプリケーションは最も使いやすい実行方法です。

1. 仮想環境が有効になっていることを確認します
2. 以下のコマンドを実行してGUIアプリを起動:

```bash
python gui_app.py
```

3. GUIが表示されたら、以下の手順で操作します:
   - 「参照」ボタンをクリックしてCSVまたはPDFファイルを選択
   - 必要に応じて分析設定を入力（目的変数、日付列など）
   - 「分析開始」ボタンをクリック
   - 分析の進行状況がログエリアに表示されます
   - 分析完了後、「出力フォルダを開く」ボタンでレポートを確認

### 方法2: コマンドラインで実行（上級者向け）

コマンドラインでより詳細なオプションを指定して実行できます。

```bash
# 基本的な使い方
python run_analysis.py <データファイルのパス> [オプション]

# オプション:
# --target    : 目的変数（予測対象）の列名
# --output    : 分析結果の出力ディレクトリ
# --api_key   : OpenAI APIキー
# --drop      : 除外する列名（スペース区切り）
# --date      : 日付として処理する列名（スペース区切り）
```

実行例:

```bash
# CSV分析の基本例
python run_analysis.py data/sales_data.csv --target revenue

# 日付列と除外列を指定
python run_analysis.py data/customer_data.csv --target churn --date signup_date last_login --drop id email

# PDFからデータを抽出して分析
python run_analysis.py reports/quarterly_report.pdf --target profit

# 出力先を指定
python run_analysis.py data/housing.csv --target price --output C:/Users/YourName/Analysis
```

### 方法3: Pythonスクリプトから実行（開発者向け）

自作のPythonスクリプトから直接利用することもできます。

```python
# example_script.py
from auto_data_analysis import AutoDataAnalysis

# 分析インスタンスの作成
analyzer = AutoDataAnalysis(
    api_key="your_openai_api_key",  # APIキー（任意）
    output_dir="~/Downloads/analysis_results"  # 出力先（任意）
)

# データ読み込みと分析の実行
analyzer.load_data("your_data.csv")
analyzer.analyze_data()

# 前処理とモデル構築
analyzer.preprocess_data(target_col="price", date_cols=["listing_date"])
analyzer.visualize_data()
analyzer.train_model()

# モデルの保存
analyzer.save_model("housing_model.pkl")

# 新しいデータに対する予測
new_data = pd.read_csv("new_houses.csv")
predictions = analyzer.predict(new_data)
```

## 詳細設定

### OpenAI APIキーの設定

AIインサイト機能を使うには、以下のいずれかの方法でAPIキーを設定します:

1. 環境変数として設定:
   ```bash
   # Windowsの場合
   set OPENAI_API_KEY=your_api_key_here
   
   # macOS/Linuxの場合
   export OPENAI_API_KEY=your_api_key_here
   ```

2. GUIアプリの「OpenAI APIキー」フィールドに直接入力

3. コマンドラインオプションで指定:
   ```bash
   python run_analysis.py data.csv --api_key your_api_key_here
   ```

### 出力ディレクトリの変更

デフォルトでは、分析結果はダウンロードフォルダの `data_analysis_results` ディレクトリに保存されます。変更するには:

1. GUIアプリの「出力ディレクトリ」フィールドで指定

2. コマンドラインオプションで指定:
   ```bash
   python run_analysis.py data.csv --output /path/to/custom/directory
   ```

3. Pythonコードで指定:
   ```python
   analyzer = AutoDataAnalysis(output_dir="/path/to/custom/directory")
   ```

## トラブルシューティング

### 一般的な問題と解決策

1. **ライブラリのインポートエラー**:
   - 仮想環境が有効化されているか確認
   - `pip install -r requirements.txt` で必要なライブラリを再インストール

2. **PDFからのテーブル抽出失敗**:
   - PDFのテキストレイヤーが正しく埋め込まれているか確認
   - 単純なテーブル構造のPDFで試す
   - PDFをCSVに手動で変換して使用

3. **メモリエラー**:
   - 大きなデータセットを扱う場合はサンプリングを検討
   - 不要な列を除外
   - より大きなメモリを持つ環境で実行

4. **AIインサイト生成エラー**:
   - APIキーが正しいか確認
   - インターネット接続を確認
   - OpenAIの利用制限や課金状況を確認

### 実行中のエラーログの確認

GUIアプリケーションでは、エラーメッセージがログエリアに表示されます。より詳細なエラー情報を確認するには、コマンドラインから実行してください。

## サポートとフィードバック

問題が解決しない場合や質問がある場合は、以下の方法でサポートを受けることができます:

- GitHubのIssueを作成
- プロジェクトのディスカッションフォーラムに投稿
- メンテナーに直接連絡

フィードバックや改善提案も歓迎します！

-----------------------
# OpenAI APIキーの設定方法

OpenAI APIキーを自動データ分析ツールで使用するには、以下の3つの方法があります。APIキーはAI分析インサイトを生成するために使用されます。

## 方法1: GUIアプリケーションで直接入力する

1. `python gui_app.py` でGUIアプリケーションを起動します
2. 「分析設定」セクションの「OpenAI APIキー（任意）」欄に、APIキーを直接入力します
3. APIキーは「*****」で表示され、安全に保管されます
4. GUIセッション中のみ有効で、アプリケーションを終了すると保存されません

## 方法2: 環境変数として設定する

この方法が最も安全で推奨される方法です。

**Windowsの場合:**

コマンドプロンプトで以下を実行:
```
set OPENAI_API_KEY=your_api_key_here
```

または、システムの環境変数設定から永続的に設定することもできます:
1. 「コントロールパネル」→「システム」→「システムの詳細設定」
2. 「環境変数」ボタンをクリック
3. 「新規」ボタンをクリックして変数名に `OPENAI_API_KEY`、値にAPIキーを入力

**macOS/Linuxの場合:**

ターミナルで以下を実行:
```
export OPENAI_API_KEY=your_api_key_here
```

永続的に設定するには、`.bashrc` や `.zshrc` に追加:
```
echo 'export OPENAI_API_KEY=your_api_key_here' >> ~/.bashrc
source ~/.bashrc
```

## 方法3: コマンドラインオプションで指定する

コマンドラインからツールを実行する場合、`--api_key` オプションを使用できます:

```
python run_analysis.py data.csv --target price --api_key your_api_key_here
```

## APIキーの取得方法

まだOpenAI APIキーをお持ちでない場合:

1. [OpenAIウェブサイト](https://platform.openai.com/)にアクセスしてアカウントを作成
2. ログイン後、ダッシュボードから「API Keys」セクションに移動
3. 「Create new secret key」をクリックして新しいAPIキーを生成
4. 生成されたキーをコピーして安全に保管（このキーは一度だけ表示されます）

## 注意事項

- APIキーは秘密情報です。公開リポジトリやコードにハードコーディングしないでください
- APIの使用には料金がかかる場合があります。OpenAIの料金ページで詳細を確認してください
- APIキーが設定されていなくても、ツールの基本機能は使用できます（AIインサイト機能のみ制限されます）

キーを設定した後、分析実行時に「AI分析インサイトを生成」オプションがチェックされていることを確認してください。

-------------------
自動データ分析ツールの画面で目的変数などの入力項目の正しい使い方を説明します：

## 目的変数の入力方法

1. **目的変数（予測対象）** 
   - これは「何を予測したいか」を指定するフィールドです
   - パン屋のデータでは、以下のような値を入力できます：
     - `売上金額` - 各商品の売上金額を予測
     - `販売数` - 各商品の販売数を予測

2. **日付列**
   - 日付として処理される列の名前を入力します
   - パン屋のデータでは `日付` と入力します
   - 複数の日付列がある場合はカンマで区切ります

3. **除外列**
   - 分析から除外したい列の名前を入力します
   - 例えば、以下のような列を除外できます：
     - `イベント` - イベント情報が少ない場合
     - ID列や参照用の列など

## 具体例

パン屋のデータを分析する場合の入力例：

- **目的変数**: `売上金額`
- **日付列**: `日付`
- **除外列**: （必要に応じて）

## 実行のコツ

1. **まずは全てのデータを見る分析**:
   - 目的変数を指定せずに実行すると、データの概要分析のみが行われます
   - これにより、データの特性を把握してから予測モデルを構築できます

2. **目的変数を指定した分析**:
   - 目的変数を指定すると、予測モデルが構築されます
   - 例：`売上金額`を予測するモデルを構築

3. **異なる目的変数での分析**:
   - 一度分析した後、異なる目的変数で再度分析できます
   - 例：`販売数`を予測するモデルを構築

## 特に重要な点

- **列名の正確な入力**: 列名は大文字・小文字や空白も正確に入力する必要があります
- **データの前処理**: 日付列を指定すると、自動的に年、月、日、曜日などの特徴量に変換されます
- **カテゴリデータ**: 「商品名」や「天気」などのカテゴリデータは自動的に適切に処理されます

これらの情報を参考に、画面で適切に情報を入力することで、データ分析ツールを最大限に活用できます。

--------------------------------
★実行CMD1UI出す検索

(data_analysis_env) PS hogehoge\20250330_datayosoku> python gui_app.py

★実行コマンド２　レポート作成だけするもの
(data_analysis_env) PS C:\Users\yukik\Desktop\20250330_datayosoku> python enhance_visualization.py ファイルpath指定
--------------------------------


