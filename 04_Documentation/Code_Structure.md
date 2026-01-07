# Code Structure Documentation

This document explains the structure and organization of the analysis code.

---

## 📂 Repository Organization

```
ORGANIZED_PAPER_CODE/
│
├── 01_Final_Analysis_Code/
│   ├── Final_Paper_Analysis_10Tables_5Figures.ipynb    # ⭐ Main file - USE THIS
│   ├── BACKUP_Original_fincode_formerged_2023-12-07.ipynb
│   ├── BACKUP_Development_v1_2023-11-09.ipynb
│   └── BACKUP_Development_v2_2023-11-09.ipynb
│
├── 02_Paper_Results/
│   ├── figures/          # 5 figure files (docx)
│   ├── tables/           # 10 table files (docx)
│   └── output_data/      # CSV result files
│
├── 03_Source_Data/
│   ├── 코스피 자기주식취득및처분.xlsx    # KOSPI raw data
│   ├── 코스닥 자기주식취득및처분.xlsx    # KOSDAQ raw data
│   └── final_data_summary.csv
│
└── 04_Documentation/
    ├── README.md                      # Overview
    ├── Code_Structure.md             # This file
    └── Paper_Output_Mapping.md       # Detailed mapping
```

---

## 🎯 Main Analysis File

### Final_Paper_Analysis_10Tables_5Figures.ipynb

**Structure**: 52 cells (41 code + 11 markdown headers)

#### Cell Organization:

```
Section 1: Data Loading
├── Cell 0: Load KOSPI/KOSDAQ data
│
Section 2: Descriptive Statistics (Table 1)
├── Cells 2-4: Data exploration
└── Cell 5: Generate Table 1
│
Section 3: Data Preprocessing
├── Cells 9-10: Remove missing/unknown values
├── Cell 11: One-hot encoding
└── Cells 12-13: Verify cleaning
│
Section 4: Behavior Analysis (Table 2)
└── Cell 16: Group by behavior, calculate Tobin Q → Table 2
│
Section 5: Feature Selection & Scaling
├── Cells 18-19: Select 6 features
└── Cells 20-21: StandardScaler
│
Section 6: K-Means Clustering (Figures 1-2, Table 3)
├── Cell 22: Elbow plot → Figure 1
├── Cell 23: Silhouette score → Figure 2
├── Cells 24-27: Fit KMeans(k=3)
└── Cell 28: Cluster means → Table 3
│
Section 7: Cluster-Behavior Analysis (Table 4, Figure 3)
└── Cell 33: Cross-tabulation → Table 4, Figure 3
│
Section 8: ML Models (Table 5, Figure 5)
├── Cells 38, 42, 43: Prepare data, split train/test
├── Cells 47-48: Train 4 models
└── Cell 49: Evaluate → Table 5, Figure 5
│
Section 9: Feature Importance (Tables 6-8, Figure 4)
├── Cells 57-58: Train by behavior
├── Cell 59: FI for Burned → Table 6
├── Cell 60: FI for Disposed_Slower → Table 7
├── Cell 61: FI for Long-term Holding → Table 8
└── Cell 62: Visualize → Figure 4
│
Section 10: Value Judgment (Table 10)
├── Cells 99, 106, 108: Define criteria, train models
└── Cell 128: Frequencies → Table 10
│
Section 11: Overall Feature Importance (Table 9)
└── Cells 135-136: Aggregate and rank → Table 9
```

---

## 🔄 Data Flow

### 1. Data Loading & Cleaning
```
Raw Excel files
    ↓
Load & merge (Cell 0)
    ↓
Remove missing values (Cell 9)
    ↓
Remove 'Unknown' behaviors (Cell 10)
    ↓
One-hot encoding (Cell 11)
    ↓
Clean dataset
```

### 2. Feature Engineering
```
Clean dataset
    ↓
Select 6 features (Cell 18)
    ↓
StandardScaler (Cell 20)
    ↓
Scaled feature matrix X
```

### 3. Clustering Analysis
```
Scaled features X
    ↓
Elbow method (Cell 22) → Figure 1
    ↓
Silhouette score (Cell 23) → Figure 2
    ↓
KMeans(k=3) (Cell 24)
    ↓
Cluster labels
    ↓
Cluster means (Cell 28) → Table 3
```

### 4. ML Pipeline
```
Scaled features X + Cluster labels y
    ↓
Create binary labels per cluster (Cell 42)
    ↓
Train/Test split 70/30 (Cell 43)
    ↓
Train 4 models (Cell 48):
    - AdaBoost
    - XGBoost
    - Gradient Boosting
    - Random Forest
    ↓
Evaluate accuracy (Cell 49) → Table 5, Figure 5
    ↓
Extract feature importance (Cells 57-62) → Tables 6-8, Figure 4
```

### 5. Value Judgment
```
Features + Behavior labels
    ↓
Define value criteria (Cell 99)
    ↓
Create judgment labels (Cell 106)
    ↓
Train classification (Cell 108)
    ↓
Frequency distribution (Cell 128) → Table 10
```

---

## 🧩 Code Dependencies

### Cell Dependencies:

```
Cell 0 (Load data)
    ↓
Cells 2-5 (Table 1) ← Independent, can run early
    ↓
Cells 9-13 (Preprocessing) ← Required for all downstream
    ↓
Cell 16 (Table 2) ← Can run after preprocessing
    ↓
Cells 18-21 (Feature scaling) ← Required for clustering & ML
    ↓
Cells 22-28 (Clustering) ← Generates cluster labels
    ↓  ↓
    ↓  Cell 33 (Table 4, Figure 3) ← Needs cluster labels
    ↓
Cells 38-49 (ML models) ← Needs cluster labels & scaled features
    ↓
Cells 57-62 (Feature importance) ← Needs ML models trained
    ↓
Cells 99-128 (Value judgment) ← Can run in parallel with FI
    ↓
Cells 135-136 (Table 9) ← Aggregates all FI results
```

### Key Dependencies:
- **All sections** depend on: Cell 0 (data loading)
- **Clustering & ML** depend on: Cells 9-13 (preprocessing) + Cells 18-21 (scaling)
- **Feature importance** depends on: Clustering results (cluster labels)
- **Table 9** depends on: All feature importance analysis (Cells 57-62)

---

## 📦 Key Variables

### Data Variables:
```python
# Main dataset (after Cell 0)
df: pd.DataFrame  # Merged KOSPI + KOSDAQ data

# After preprocessing (Cell 13)
df_clean: pd.DataFrame  # Cleaned dataset

# Feature matrix (Cell 19)
X: np.ndarray  # Shape: (n_samples, 6)

# Scaled features (Cell 20)
X_scaled: np.ndarray  # StandardScaler applied
```

### Target Variables:
```python
# Cluster labels (Cell 25)
cluster_labels: np.ndarray  # Values: 0, 1, 2

# Behavior types (Cell 16)
behavior: pd.Series  # Values: 'Burned', 'Disposed_Slower', 'Long-term Holding'

# Value judgment (Cell 106)
value_judgment: pd.Series  # Values: 0 (undervalued), 1 (fairly valued)
```

### Model Variables:
```python
# ML models (Cell 47)
models = {
    'AdaBoost': AdaBoostClassifier(random_state=0),
    'XGBoost': XGBClassifier(random_state=0),
    'GradientBoosting': GradientBoostingClassifier(random_state=0),
    'RandomForest': RandomForestClassifier(random_state=0)
}
```

---

## 🛠️ Key Functions & Methods

### Data Processing:
```python
# Load data
pd.read_excel()
pd.concat()

# Cleaning
df.dropna()
df[df['Behavior'] != 'Unknown']
pd.get_dummies()

# Feature selection
df[['자산총계', '부채총계', ...]]
```

### Clustering:
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=0)
cluster_labels = kmeans.fit_predict(X_scaled)

score = silhouette_score(X_scaled, cluster_labels)
```

### Machine Learning:
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=0, stratify=y
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
```

### Feature Importance:
```python
# For tree-based models
feature_importance = model.feature_importances_

# Create importance dataframe
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)
```

---

## 📊 Output Generation

### Tables (DataFrames):
```python
# Table 1: Descriptive statistics
df.describe()

# Table 2: Behavior analysis
df.groupby('Behavior')['Tobin_Q'].mean()

# Table 3: Cluster means
df.groupby('Cluster')[features].mean()

# Table 4: Cross-tabulation
pd.crosstab(df['Cluster'], df['Behavior'])

# Table 5: Model performance
pd.DataFrame({'Model': [...], 'Accuracy': [...]})

# Tables 6-8: Feature importance by behavior
importance_df.head(10)

# Table 9: Overall feature importance
all_importance_df.groupby('feature').mean().sort_values()

# Table 10: Value judgment frequencies
df['Value_Judgment'].value_counts()
```

### Figures (Matplotlib/Seaborn):
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Figure 1: Elbow plot
plt.plot(k_values, inertias)
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')

# Figure 2: Silhouette score
plt.bar(['k=3'], [silhouette_score])

# Figure 3: Cluster-Behavior distribution
sns.countplot(x='Cluster', hue='Behavior', data=df)

# Figure 4: Feature importance
plt.barh(features, importances)

# Figure 5: Model performance
plt.bar(model_names, accuracies)
```

---

## 🔧 Configuration Parameters

### Global Settings:
```python
random_state = 0          # For reproducibility
test_size = 0.3           # 70/30 train/test split
stratify = True           # Stratified sampling
```

### Clustering:
```python
n_clusters = 3            # Optimal k
max_k = 10                # For elbow plot
```

### Feature Selection:
```python
n_features = 6            # Selected features
scaling_method = 'StandardScaler'
```

---

## 📝 Code Style & Conventions

### Variable Naming:
- DataFrames: `df`, `df_clean`, `df_train`
- Arrays: `X`, `y`, `X_scaled`
- Models: `model`, `clf`, or descriptive names like `ada_boost`
- Results: `accuracy`, `importance`, `predictions`

### Comments:
- Section headers: Clear markdown cells
- Code comments: For complex operations only
- Output descriptions: Before each table/figure generation

### File Saving:
- CSV files: `df.to_csv('filename.csv', index=False)`
- Figures: Manual save from notebook (not automated in code)
- Tables: Manual copy to Word (not automated in code)

---

## ⚡ Performance Notes

### Execution Time:
- Data loading: ~5 seconds
- Preprocessing: ~2 seconds
- Clustering: ~10 seconds
- ML model training: ~30 seconds (all 4 models)
- Feature importance: ~20 seconds
- **Total**: ~2-3 minutes for full notebook execution

### Memory Usage:
- Raw data: ~50 MB
- Processed data: ~30 MB
- Models: ~10 MB
- **Total**: ~100 MB peak memory

---

## 🐛 Common Issues & Solutions

### Issue 1: Data file not found
```python
# Solution: Check data path in Cell 0
# Adjust to: '../03_Source_Data/filename.xlsx'
```

### Issue 2: Korean encoding errors
```python
# Solution: Ensure UTF-8 encoding
pd.read_excel('file.xlsx', encoding='utf-8')
```

### Issue 3: Missing library
```bash
# Solution: Install required packages
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

### Issue 4: Different results
```python
# Solution: Check random_state is set to 0 in all models
# Verify data cleaning steps are consistent
```

---

## 📚 Additional Resources

### Related Files:
- `Paper_Output_Mapping.md` - Detailed cell-to-output mapping
- `README.md` - Project overview and quick start guide

### Python Documentation:
- scikit-learn: https://scikit-learn.org/
- XGBoost: https://xgboost.readthedocs.io/
- Pandas: https://pandas.pydata.org/

---

**Document Version**: 1.0
**Last Updated**: 2026-01-07
**Corresponds to**: Final_Paper_Analysis_10Tables_5Figures.ipynb (52 cells)
