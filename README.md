# Task 3 — Exploratory Data Analysis (EDA)

A Python script that performs a complete exploratory data analysis on any CSV dataset: structural overview, missing values, duplicates, summary statistics, univariate and bivariate analysis, correlation analysis, and outlier detection — with all charts saved automatically as image files.

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn
```

## Setup

1. Open `task3_eda.py`.
2. Update the `DATA_PATH` variable near the top of the file to point to your CSV dataset:

```python
DATA_PATH = r"C:\Users\ADMIN\Downloads\customer_sales.csv"
```

3. (Optional) Change `OUTPUT_DIR` if you want charts saved to a different folder. Default is `eda_outputs`.

## Usage

Run the script from the command line:

```bash
python task3_eda.py
```

All console output (statistics, summaries, outlier counts) prints directly to the terminal. All charts are saved as `.png` files inside the `eda_outputs/` folder.

## What the Script Does

| Step | Section | Output |
|------|---------|--------|
| 1 | Basic Information | Shape, dtypes, head/tail rows, `df.info()` |
| 2 | Missing Values | Count + percentage per column, heatmap (if any missing) |
| 3 | Duplicate Rows | Count of duplicate rows |
| 4 | Summary Statistics | `describe()` for numeric and categorical columns |
| 5 | Univariate Analysis | Histogram + boxplot per numeric column, bar chart of top categories per categorical column |
| 6 | Correlation Analysis | Correlation matrix + heatmap (numeric columns) |
| 7 | Outlier Detection | IQR method, prints outlier count and bounds per numeric column |
| 8 | Bivariate Analysis | Pairplot across numeric columns (sampled to 500 rows max for speed) |

## Output Files

After running, `eda_outputs/` will contain:

- `missing_values_heatmap.png` (only if missing values exist)
- `dist_<column>.png` — distribution histogram per numeric column
- `box_<column>.png` — boxplot per numeric column
- `cat_<column>.png` — bar chart per categorical column
- `correlation_heatmap.png`
- `pairplot.png`

## Notes

- The script expects the target file to be a CSV. If you're working with Excel files, convert to CSV first or adjust `load_data()` to use `pd.read_excel()`.
- Categorical bar charts only show the top 15 categories per column to keep plots readable.
- The pairplot samples up to 500 rows to avoid slow rendering on large datasets.
- Tested with pandas (string-dtype-aware) and current seaborn versions — no deprecation warnings.
