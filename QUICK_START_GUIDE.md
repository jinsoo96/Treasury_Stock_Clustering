# Quick Start Guide

**Get started with the Stock Buyback Analysis code in 5 minutes.**

---

## 🚀 Quick Start

### Step 1: Open the Notebook
```bash
cd ORGANIZED_PAPER_CODE/01_Final_Analysis_Code
jupyter notebook Final_Paper_Analysis_10Tables_5Figures.ipynb
```

### Step 2: Run All Cells
```
Kernel → Restart & Run All
```

### Step 3: Check Results
- Tables 1-10: Generated in cell outputs
- Figures 1-5: Displayed inline
- CSV files: Saved to `02_Paper_Results/output_data/`

**Done! All paper outputs reproduced in ~3 minutes.**

---

## 📋 Prerequisites

### Required Software:
```bash
# Python 3.8+
python --version

# Jupyter
pip install jupyter

# Required packages
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

### Required Data:
Files should be in `03_Source_Data/`:
- `KOSPI_Stock_Buyback.xlsx`
- `KOSDAQ_Stock_Buyback.xlsx`

---

## 📊 What Gets Generated

### 10 Tables:
1. Descriptive statistics
2. Behavior analysis (Tobin Q)
3. Cluster summary (k=3)
4. Cluster-Behavior cross-tab
5. ML model performance
6-8. Feature importance (by behavior)
9. Overall feature importance
10. Value judgment classification

### 5 Figures:
1. Elbow plot (optimal k)
2. Silhouette score
3. Cluster-Behavior distribution
4. Feature importance charts
5. Model performance comparison

### Output Files:
- `cluster4_summary.csv`
- `labelfeature.csv`
- `label_freq.csv`
- `Final_Processed_Data_2024-01-10.csv`

---

## 🎯 Key Code Sections

| Section | What It Does | Outputs |
|---------|--------------|---------|
| 1. Data Loading | Load KOSPI/KOSDAQ data | - |
| 2. Descriptive Stats | Summary statistics | Table 1 |
| 3. Preprocessing | Clean data | - |
| 4. Behavior Analysis | Tobin Q by behavior | Table 2 |
| 5. Feature Selection | Select & scale 6 features | - |
| 6. Clustering | K-Means (k=3) | Figs 1-2, Table 3 |
| 7. Cluster-Behavior | Cross-tabulation | Table 4, Fig 3 |
| 8. ML Models | Train 4 models | Table 5, Fig 5 |
| 9. Feature Importance | By behavior type | Tables 6-8, Fig 4 |
| 10. Value Judgment | Undervalued classification | Table 10 |
| 11. Overall FI | Final summary | Table 9 |

---

## 🔧 Common Modifications

### Change number of clusters:
```python
# Cell 24: Change k value
kmeans = KMeans(n_clusters=4, random_state=0)  # Try k=4 instead
```

### Add more features:
```python
# Cell 18: Expand feature list
features = [
    '자산총계(요약)(백만원)',
    '부채총계(요약)(백만원)',
    '자본총계(요약)(백만원)',
    '당기순이익(요약)(백만원)',
    'ROE(자기자본이익률)',
    '구분',
    'YOUR_NEW_FEATURE'  # Add here
]
```

### Change train/test split:
```python
# Cell 43: Adjust test_size
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=0, stratify=y
)  # Changed from 0.3 to 0.2
```

### Try different ML models:
```python
# Cell 47: Add more models
from sklearn.linear_model import LogisticRegression

models['LogisticRegression'] = LogisticRegression(random_state=0)
```

---

## 📁 File Structure

```
ORGANIZED_PAPER_CODE/
├── 01_Final_Analysis_Code/
│   └── Final_Paper_Analysis_10Tables_5Figures.ipynb  ← START HERE
│
├── 02_Paper_Results/
│   ├── figures/        ← Generated figures
│   ├── tables/         ← Generated tables
│   └── output_data/    ← CSV outputs
│
├── 03_Source_Data/
│   └── *.xlsx          ← Raw data files
│
└── 04_Documentation/
    ├── README.md               ← Full documentation
    ├── Code_Structure.md       ← Code details
    ├── Paper_Output_Mapping.md ← Cell-to-output mapping
    └── QUICK_START_GUIDE.md    ← This file
```

---

## ⚠️ Troubleshooting

### Problem: Data file not found
```python
# Solution: Check path in Cell 0
# Update to: '../03_Source_Data/filename.xlsx'
```

### Problem: Module not found
```bash
# Solution: Install missing package
pip install package_name
```

### Problem: Different results each run
```python
# Solution: Verify random_state=0 in all models
# Check: Cells 24, 43, 47, 48
```

### Problem: Memory error
```python
# Solution: Reduce dataset size or close other applications
# Or use larger RAM machine
```

---

## 📖 Learn More

### Documentation:
- Full README: `04_Documentation/README.md`
- Code structure: `04_Documentation/Code_Structure.md`
- Output mapping: `04_Documentation/Paper_Output_Mapping.md`

### Key Concepts:
- **K-Means Clustering**: Groups firms into 3 clusters based on financial features
- **Elbow Method**: Finds optimal number of clusters
- **Feature Importance**: Identifies which variables best predict behaviors
- **Behavior Types**: Burned, Disposed_Slower, Long-term Holding

---

## 🎓 Understanding the Analysis

### Research Question:
*How do Korean firms behave after stock buybacks, and what factors predict their behavior?*

### Methodology:
1. **Classify** firms by buyback behavior (4 types)
2. **Cluster** firms by financial characteristics (k=3)
3. **Predict** behavior using ML models (4 models)
4. **Identify** key features driving behavior

### Key Finding:
**Total Assets** is the most important predictor of buyback behavior.

---

## ⏱️ Expected Runtime

| Task | Time |
|------|------|
| Data loading | ~5 sec |
| Preprocessing | ~2 sec |
| Clustering | ~10 sec |
| ML models | ~30 sec |
| Feature importance | ~20 sec |
| **Total** | **~3 min** |

---

## 💡 Tips

1. **Run sequentially**: Don't skip cells - they depend on each other
2. **Check outputs**: Verify tables and figures are generated correctly
3. **Save results**: Copy tables/figures to separate files if needed
4. **Modify carefully**: Changing early cells affects all downstream results
5. **Use random_state=0**: Ensures reproducible results

---

## 📞 Need Help?

1. Check `04_Documentation/README.md` for detailed information
2. Review `Code_Structure.md` for code dependencies
3. Consult `Paper_Output_Mapping.md` for specific table/figure generation

---

## ✅ Checklist

Before running the notebook:
- [ ] Python 3.8+ installed
- [ ] Jupyter installed
- [ ] Required packages installed (pandas, numpy, sklearn, xgboost, matplotlib, seaborn)
- [ ] Data files in `03_Source_Data/`
- [ ] Notebook opened: `Final_Paper_Analysis_10Tables_5Figures.ipynb`

Ready to run:
- [ ] Kernel → Restart & Run All
- [ ] Wait ~3 minutes
- [ ] Check all 52 cells executed successfully
- [ ] Verify tables and figures generated
- [ ] Review output CSV files in `02_Paper_Results/output_data/`

---

**Happy analyzing! 🎉**

**Last Updated**: 2026-01-07
**Version**: 1.0
