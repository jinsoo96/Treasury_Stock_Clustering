# Final Analysis Code

This folder contains the final analysis code that generates all paper results.

---

## Files in This Folder

### stock_buyback_analysis.ipynb
**THE MAIN FILE - Use this to reproduce all paper results.**

- **Purpose**: Generate all 10 tables and 5 figures for the paper
- **Structure**: 52 cells (41 code + 11 markdown headers)
- **Status**: Cleaned, documented, ready to run
- **Execution time**: ~3 minutes
- **Outputs**: 10 tables, 5 figures, 4 CSV files

**This is the file you should use for:**
- Reproducing paper results
- Understanding the analysis workflow
- Modifying or extending the analysis
- Teaching or presenting the methodology

---

## Which File Should I Use?

| Task | Recommended File | Reason |
|------|------------------|--------|
| **Reproduce paper results** | stock_buyback_analysis.ipynb | ✅ Clean, documented, complete |
| **Understand methodology** | stock_buyback_analysis.ipynb | ✅ Clear structure, English headers |
| **Modify analysis** | stock_buyback_analysis.ipynb | ✅ Minimal cells, easy to change |

**Bottom line: Use `stock_buyback_analysis.ipynb` for everything.**

---

## What Does the Main File Generate?

### Tables (10):
1. Descriptive statistics
2. Behavior analysis by Tobin Q
3. K-Means cluster summary (k=3)
4. Cluster-Behavior cross-tabulation
5. ML model performance comparison
6. Feature importance: Burned behavior
7. Feature importance: Disposed_Slower behavior
8. Feature importance: Long-term Holding behavior
9. Overall feature importance summary
10. Value judgment classification results

### Figures (5):
1. Elbow plot (optimal cluster selection)
2. Silhouette score (k=3)
3. Cluster-Behavior distribution
4. Feature importance by behavior (bar charts)
5. ML model performance comparison

### CSV Output Files:
- `cluster4_summary.csv` - Cluster mean values
- `labelfeature.csv` - Feature importance details
- `label_freq.csv` - Behavior and value judgment frequencies
- `Final_Processed_Data_2024-01-10.csv` - Complete processed dataset

---

## Quick Start

### 1. Open the notebook:
```bash
jupyter notebook stock_buyback_analysis.ipynb
```

### 2. Run all cells:
```
Kernel → Restart & Run All
```

### 3. Check results:
- Tables appear in cell outputs
- Figures display inline
- CSV files saved to `../02_Paper_Results/output_data/`

**Done in ~3 minutes!**

---

## Code Structure

The main notebook has 11 clearly marked sections:

```
1. Data Loading
   └── Load KOSPI/KOSDAQ stock buyback data

2. Descriptive Statistics → Table 1
   └── Summary stats for all variables

3. Data Preprocessing
   └── Clean, encode, prepare data

4. Behavior Analysis → Table 2
   └── Tobin Q analysis by behavior type

5. Feature Selection & Scaling
   └── Select 6 features, apply StandardScaler

6. K-Means Clustering → Figures 1-2, Table 3
   └── Elbow plot, silhouette score, cluster means

7. Cluster-Behavior Analysis → Table 4, Figure 3
   └── Cross-tabulation and visualization

8. ML Model Training → Table 5, Figure 5
   └── 4 models: AdaBoost, XGBoost, GB, RF

9. Feature Importance → Tables 6-8, Figure 4
   └── By behavior: Burned, Disposed_Slower, Long-term Holding

10. Value Judgment → Table 10
    └── Undervalued vs. Fairly valued classification

11. Overall Feature Importance → Table 9
    └── Aggregate and rank all features
```

Each section has:
- **Markdown header**: Explains purpose and outputs
- **Code cells**: Clean, essential operations only
- **Outputs**: Tables, figures, or data processing results

---

## Key Configurations

### Clustering:
```python
n_clusters = 3          # Optimal k
random_state = 0        # For reproducibility
scaler = StandardScaler()
```

### ML Models:
```python
models = {
    'AdaBoost': AdaBoostClassifier(random_state=0),
    'XGBoost': XGBClassifier(random_state=0),
    'GradientBoosting': GradientBoostingClassifier(random_state=0),
    'RandomForest': RandomForestClassifier(random_state=0)
}
```

### Train/Test Split:
```python
test_size = 0.3         # 70/30 split
random_state = 0
stratify = True         # Stratified sampling
```

### Features (6):
1. Total Assets (자산총계)
2. Total Liabilities (부채총계)
3. Total Equity (자본총계)
4. Net Income (당기순이익)
5. ROE (자기자본이익률)
6. Market Classification (구분: KOSPI/KOSDAQ)

---

## File Information

| Metric | Final |
|--------|-------|
| **Cells** | 52 |
| **Code cells** | 41 |
| **English headers** | ✅ Yes |
| **Clean structure** | ✅ Yes |
| **Complete outputs** | ✅ Yes |
| **Date** | 2026-01-07 |
| **Recommended** | ⭐⭐⭐⭐⭐ |

---

## ⚠️ Important Notes

### 1. Data Paths
Check data file paths in Cell 0:
```python
# May need to adjust to:
df_kospi = pd.read_excel('../03_Source_Data/KOSPI_Stock_Buyback.xlsx')
df_kosdaq = pd.read_excel('../03_Source_Data/KOSDAQ_Stock_Buyback.xlsx')
```

### 2. Dependencies
All sections depend on previous sections. Do not skip cells!

### 3. Random State
All models use `random_state=0` for reproducibility.

### 4. Korean Column Names
Some column names remain in Korean (original data format).

---

## Troubleshooting

### Problem: Cells fail to execute
- **Solution**: Run cells sequentially from Cell 0
- **Cause**: Missing dependencies from skipped cells

### Problem: Different results
- **Solution**: Check `random_state=0` in all models
- **Cause**: Random initialization without seed

### Problem: File not found
- **Solution**: Check data paths in Cell 0
- **Cause**: Incorrect relative path to data files

### Problem: Import error
- **Solution**: `pip install pandas numpy scikit-learn xgboost matplotlib seaborn`
- **Cause**: Missing required packages

---

## Additional Documentation

For more details, see:
- `../README.md` - Main repository overview
- `../04_Documentation/Code_Structure.md` - Detailed code structure
- `../04_Documentation/Paper_Output_Mapping.md` - Cell-to-output mapping
- `../QUICK_START_GUIDE.md` - Quick start tutorial

---

## Verification Checklist

Before using the notebook:
- [ ] Python 3.8+ installed
- [ ] Jupyter notebook installed
- [ ] Required packages installed
- [ ] Data files in `../03_Source_Data/`
- [ ] Data paths in Cell 0 verified

After running:
- [ ] All 52 cells executed successfully
- [ ] 10 tables generated (visible in outputs)
- [ ] 5 figures displayed inline
- [ ] 4 CSV files created in `../02_Paper_Results/output_data/`
- [ ] No error messages

---

**Last Updated**: 2026-01-07
**Version**: 1.0
**Status**: Production ready - Use for all paper reproduction
