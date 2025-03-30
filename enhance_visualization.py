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
    print("Note: japanize-matplotlib is not installed, Japanese characters may not display correctly")
    print("To install: pip install japanize-matplotlib")

def enhance_visualization_with_weekday(csv_file_path, output_dir=None):
    """
    Function to create enhanced visualizations based on date, weekday, and columns in sales data
    
    Parameters:
    -----------
    csv_file_path : str
        Path to the CSV file
    output_dir : str, optional
        Output directory (if not specified, creates 'enhanced_viz' in the current directory)
    """
    # Record start time
    start_time = datetime.now()
    
    # Setup output directory
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'enhanced_viz')
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory for enhanced visualizations: {output_dir}")
    
    # Load data
    print(f"Loading CSV file: {csv_file_path}")
    try:
        df = pd.read_csv(csv_file_path)
        print(f"Data loaded successfully: {df.shape[0]} rows x {df.shape[1]} columns")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Convert numeric columns - both English and Japanese detection
    numeric_cols = ['Number of Sales', 'Unit Price(yen)', 'Sales Amount(yen)', 
                   '販売数', '単価(円)', '売上金額(円)']
    for col in numeric_cols:
        if col in df.columns:
            # Remove commas and spaces, convert to numeric
            try:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '').str.replace(' ', ''), errors='coerce')
                print(f"Column '{col}' converted to numeric type")
            except Exception as e:
                print(f"Failed to convert column '{col}' to numeric: {e}")
    
    # Check available columns
    print("\nAvailable columns:")
    for i, col in enumerate(df.columns):
        print(f"{i+1}. {col} - Sample value: {df[col].iloc[0] if not df.empty else 'none'}")
    
    # Auto-detect date column - support both English and Japanese
    date_columns = [col for col in df.columns if any(date_term in col.lower() for date_term in ['date', '日付', '日時', '年月日'])]
    
    if date_columns:
        date_column = date_columns[0]
        print(f"\nAutomatically detected date column: {date_column}")
    else:
        print("\nCould not automatically detect date column.")
        # Default to "日付" if it exists
        if "日付" in df.columns:
            date_column = "日付"
            print(f"Using column '日付' as date column")
        else:
            date_column = input("Enter date column name: ")
    
    # Convert date to datetime
    try:
        df[date_column] = pd.to_datetime(df[date_column])
        print(f"Date column '{date_column}' converted to datetime format")
    except Exception as e:
        print(f"Error converting date: {e}")
        return None
    
    # Check if we have a Japanese or English weekday column
    has_jp_weekday = False
    has_en_weekday = False
    
    if '曜日' in df.columns:
        weekday_col = '曜日'
        # Check if weekdays are in Japanese
        unique_weekdays = df['曜日'].unique()
        if any(day in ['月', '火', '水', '木', '金', '土', '日'] for day in unique_weekdays):
            has_jp_weekday = True
            weekday_order = ['月', '火', '水', '木', '金', '土', '日']
            print("Using Japanese weekday column")
        elif any(day in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'] for day in unique_weekdays):
            has_en_weekday = True
            weekday_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            print("Using English weekday column")
        else:
            # Default to Japanese
            has_jp_weekday = True
            weekday_order = ['月', '火', '水', '木', '金', '土', '日']
            print("Using weekday column (assuming Japanese format)")
    else:
        # Create a new weekday column
        # Check if datetime index day_name is in Japanese or English
        # For safety, we'll use numeric mapping for Japanese
        weekday_map_jp = {0: '月', 1: '火', 2: '水', 3: '木', 4: '金', 5: '土', 6: '日'}
        weekday_map_en = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
        
        # Default to Japanese weekday names
        df['曜日'] = df[date_column].dt.weekday.map(weekday_map_jp)
        weekday_col = '曜日'
        weekday_order = ['月', '火', '水', '木', '金', '土', '日']
        has_jp_weekday = True
        print("Created new Japanese weekday column")
    
    # Auto-detect sales column - support both English and Japanese
    sales_columns = [col for col in df.columns if any(sales_term in col.lower() for sales_term in 
                                                     ['sales', 'revenue', 'amount', '売上', '金額', '収入'])]
    
    if sales_columns:
        sales_column = sales_columns[0]
        print(f"\nAutomatically detected sales column: {sales_column}")
    else:
        print("\nCould not automatically detect sales column.")
        # Default to "売上金額(円)" if it exists
        if "売上金額(円)" in df.columns:
            sales_column = "売上金額(円)"
            print(f"Using column '売上金額(円)' as sales column")
        else:
            sales_column = input("Enter sales column name: ")
    
    # Auto-detect quantity column - support both English and Japanese
    quantity_columns = [col for col in df.columns if any(qty_term in col.lower() for qty_term in 
                                                       ['quantity', 'number', 'amount', '販売', '数量', '個数'])]
    
    if quantity_columns:
        quantity_column = quantity_columns[0]
        print(f"\nAutomatically detected quantity column: {quantity_column}")
    else:
        print("\nCould not automatically detect quantity column. Specify manually if needed.")
        # Default to "販売数" if it exists
        if "販売数" in df.columns:
            quantity_column = "販売数"
            print(f"Using column '販売数' as quantity column")
        else:
            quantity_column = None
    
    # Auto-detect product column - support both English and Japanese
    product_columns = [col for col in df.columns if any(prod_term in col.lower() for prod_term in 
                                                      ['product', 'item', '商品', '製品'])]
    
    if product_columns:
        product_column = product_columns[0]
        print(f"\nAutomatically detected product column: {product_column}")
    else:
        print("\nCould not automatically detect product column. Specify manually if needed.")
        # Default to "商品名" if it exists
        if "商品名" in df.columns:
            product_column = "商品名"
            print(f"Using column '商品名' as product column")
        else:
            product_column = None
    
    # Detect weather column - support both English and Japanese
    weather_column = None
    for weather_term in ['Weather', '天気', '気象']:
        if weather_term in df.columns:
            weather_column = weather_term
            print(f"\nDetected weather column: {weather_column}")
            break
    
    # Begin visualization
    print("\nExecuting enhanced visualizations...")
    
    # Create report file - choose language based on detected columns
    report_title = "曜日別データ可視化レポート" if has_jp_weekday else "Weekday Data Visualization Report"
    generated_text = "生成日時" if has_jp_weekday else "Generated on"
    data_file_text = "データファイル" if has_jp_weekday else "Data file"
    records_text = "レコード数" if has_jp_weekday else "Records"
    fields_text = "項目数" if has_jp_weekday else "Fields"
    
    report_file = os.path.join(output_dir, "visualization_report.md")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"# {report_title}\n\n")
        f.write(f"**{generated_text}**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**{data_file_text}**: {os.path.basename(csv_file_path)}\n")
        f.write(f"**{records_text}**: {df.shape[0]} rows\n")
        f.write(f"**{fields_text}**: {df.shape[1]} columns\n\n")
    
    # ========== 1. Sales by weekday ==========
    try:
        plt.figure(figsize=(12, 6))
        weekday_sales = df.groupby(weekday_col)[sales_column].sum().reindex(weekday_order)
        
        # Ensure numeric values
        weekday_sales = pd.to_numeric(weekday_sales, errors='coerce')
        
        # Create graph
        ax = sns.barplot(x=weekday_sales.index, y=weekday_sales.values, color='steelblue')
        
        # Title in appropriate language
        title_by_weekday = f"曜日別の{sales_column}合計" if has_jp_weekday else f"Total {sales_column} by Weekday"
        weekday_label = "曜日" if has_jp_weekday else "Weekday"
        
        plt.title(title_by_weekday, fontsize=16)
        plt.xlabel(weekday_label, fontsize=14)
        plt.ylabel(sales_column, fontsize=14)
        
        # Display values on bars
        for i, v in enumerate(weekday_sales.values):
            if pd.notnull(v):  # Check for null values
                height_offset = max(v * 0.02, 1000) if v > 0 else 1000
                ax.text(i, v + height_offset, f'{int(v):,}', ha='center', fontsize=12)
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save
        output_path = os.path.join(output_dir, f'Weekday_{sales_column}.png')
        plt.savefig(output_path, dpi=300)
        print(f"Saved: Weekday_{sales_column}.png")
        
        # Add to report - in appropriate language
        report_title_text = f"## 曜日別の{sales_column}合計" if has_jp_weekday else f"## Total {sales_column} by Weekday"
        day_header = "曜日" if has_jp_weekday else "Weekday"
        amount_header = "合計金額" if has_jp_weekday else "Total Amount"
        yen_text = "円" if has_jp_weekday else "yen"
        
        with open(report_file, "a", encoding="utf-8") as f:
            f.write(f"{report_title_text}\n\n")
            f.write(f"![Weekday Sales](Weekday_{sales_column}.png)\n\n")
            f.write(f"| {day_header} | {amount_header} |\n")
            f.write("|------|----------:|\n")
            for day, value in weekday_sales.items():
                if pd.notnull(value):
                    f.write(f"| {day} | {int(value):,} {yen_text} |\n")
            f.write("\n")
    except Exception as e:
        print(f"Error during weekday sales analysis: {e}")

    # ========== Continue with other visualizations ==========
    # ... (add the rest of the visualization code here) ...
    
    print(f"\nEnhanced visualization complete! Report file: {report_file}")
    
    return df  # Return the processed dataframe

# Run from command line
if __name__ == "__main__":
    # Process command line arguments
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Tool for weekday data visualization from CSV files')
    parser.add_argument('file_path', help='Path to CSV file')
    parser.add_argument('--output', '-o', help='Output directory', default=None)
    parser.add_argument('--target', '-t', help='Sales column name (if auto-detection fails)', default=None)
    
    # Display help if no arguments
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    args = parser.parse_args()
    
    # Run visualization
    enhance_visualization_with_weekday(args.file_path, args.output)