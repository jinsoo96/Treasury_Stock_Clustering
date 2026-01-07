# Stock Buyback Analysis - Korean Market

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-clean-brightgreen.svg)](https://github.com/psf/black)
[![Status](https://img.shields.io/badge/status-production-success.svg)](https://github.com)

> **Analyzing stock buyback behavior patterns in Korean stock market (KOSPI/KOSDAQ) using machine learning and clustering techniques**

Research code repository for paper: *"Analysis of Stock Buyback Behavior in Korean Stock Market"*

---

## Overview

This repository contains the complete analysis pipeline that generates **10 tables** and **5 figures** for our research paper on stock buyback behavior in the Korean stock market.

### Key Features

- **Behavior Classification**: Categorize firms into 4 buyback behavior types
- **K-Means Clustering**: Group firms based on financial characteristics (k=3)
- **ML Prediction Models**: 4 models (AdaBoost, XGBoost, Gradient Boosting, Random Forest)
- **Feature Importance Analysis**: Identify key predictors of buyback behavior
- **Value Judgment**: Classify undervalued vs. fairly valued firms

### Quick Stats

| Metric | Value |
|--------|-------|
| **Data Points** | 9.5 MB processed data |
| **Models** | 4 ML algorithms |
| **Clusters** | 3 optimal clusters |
| **Behaviors** | 4 types identified |
| **Execution Time** | ~3 minutes |
| **Code Cells** | 41 essential cells |

---

## Sample Results

### Clustering Analysis
```
Cluster 0: Large firms with high assets
Cluster 1: Medium-sized firms with moderate ROE
Cluster 2: Small firms with varying profitability
```

### Model Performance
```
AdaBoost:          85.3% accuracy
XGBoost:           87.1% accuracy
Gradient Boosting: 86.5% accuracy
Random Forest:     84.9% accuracy
```

### Top Feature
```
 Total Assets (자산총계) - Most important predictor
```

---

##  Quick Start

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# Install dependencies
pip install -r requirements.txt
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/stock-buyback-analysis.git
cd stock-buyback-analysis

# Install required packages
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
```

### Run Analysis

```bash
# Navigate to code folder
cd 01_Final_Analysis_Code

# Launch Jupyter
jupyter notebook Final_Paper_Analysis_10Tables_5Figures.ipynb

# Run all cells: Kernel → Restart & Run All
```

** Execution Time**: ~3 minutes to generate all results

---

##  Repository Structure

```
ORGANIZED_PAPER_CODE/
│
├──  01_Final_Analysis_Code/
│   ├── stock_buyback_analysis.ipynb  ⭐ Main notebook
│   ├── Archive_Original_fincode_merged_2023-12-07.ipynb
│   ├── Archive_Development_v1_2023-11-09.ipynb
│   └── Archive_Development_v2_2023-11-09.ipynb
│
├──  02_Paper_Results/
│   ├── figures/        # 5 figures (docx format)
│   ├── tables/         # 10 tables (docx format)
│   └── output_data/    # CSV result files
│
├──  03_Source_Data/
│   ├── KOSPI_Stock_Buyback.xlsx    # KOSPI data
│   └── KOSDAQ_Stock_Buyback.xlsx   # KOSDAQ data
│
├──  04_Documentation/
│   ├── Code_Structure.md               # Detailed code documentation
│   └── Paper_Output_Mapping.md         # Cell-to-output mapping
│
├── README.md                           # This file
├── QUICK_START_GUIDE.md               # 5-minute tutorial
├── requirements.txt                    # Python dependencies
└── LICENSE                            # License information
```

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [README.md](README.md) | Project overview (this file) |
| [QUICK_START_GUIDE.md](QUICK_START_GUIDE.md) | 5-minute tutorial |
| [Code_Structure.md](04_Documentation/Code_Structure.md) | Detailed code explanation |
| [Paper_Output_Mapping.md](04_Documentation/Paper_Output_Mapping.md) | Exact cell-to-output mapping |

---

## Methodology

### 1. Data Collection
- **Source**: Korea Exchange (KRX)
- **Markets**: KOSPI & KOSDAQ
- **Period**: 2023
- **Firms**: 1000+ companies

### 2. Behavior Classification (4 Types)

| Behavior | Description |
|----------|-------------|
| **Long-term Holding** | Firms holding buyback shares for extended periods |
| **Disposed_Slower** | Firms disposing shares at a slower rate |
| **Disposed_Faster** | Firms disposing shares at a faster rate |
| **Burned** | Firms canceling (burning) buyback shares |

### 3. Feature Engineering (6 Features)

```python
features = [
    'Total Assets (자산총계)',           # Total company assets
    'Total Liabilities (부채총계)',      # Total company liabilities
    'Total Equity (자본총계)',           # Total shareholder equity
    'Net Income (당기순이익)',           # Net income
    'ROE',                               # Return on Equity
    'Market (구분)'                      # KOSPI or KOSDAQ
]
```

### 4. Analysis Pipeline

```
Raw Data → Preprocessing → Feature Selection → Clustering (K-Means, k=3)
                                                    ↓
         Value Judgment ← Feature Importance ← ML Models (4 algorithms)
```

---

## Paper Outputs

### Tables (10)

| # | Description | Key Finding |
|---|-------------|-------------|
| 1 | Descriptive statistics | Sample characteristics |
| 2 | Behavior by Tobin Q | Relationship between valuation and behavior |
| 3 | Cluster summary | 3 distinct firm groups |
| 4 | Cluster-Behavior matrix | Behavior distribution across clusters |
| 5 | Model performance | XGBoost achieves 87.1% accuracy |
| 6-8 | Feature importance | Total Assets most predictive |
| 9 | Overall feature ranking | Comprehensive feature analysis |
| 10 | Value judgment | Undervalued firm identification |

### Figures (5)

| # | Type | Description |
|---|------|-------------|
| 1 | Line plot | Elbow plot for optimal k selection |
| 2 | Score plot | Silhouette score analysis (k=3) |
| 3 | Distribution | Cluster-Behavior visualization |
| 4 | Bar chart | Feature importance by behavior type |
| 5 | Bar chart | Model performance comparison |

---

## Code Example

```python
# Load and preprocess data
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=0)
clusters = kmeans.fit_predict(X_scaled)

# Train ML models
from xgboost import XGBClassifier
model = XGBClassifier(random_state=0)
model.fit(X_train, y_train)

# Feature importance
importance = model.feature_importances_
```

---

## Requirements

### Python Packages

```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
```

### System Requirements

- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 50MB for code, 15MB for data
- **OS**: Windows, macOS, or Linux

---

## Key Findings

### 1. Firm Clustering
- Three distinct clusters identified based on financial characteristics
- Cluster membership correlates with buyback behavior

### 2. Behavior Predictors
- **Total Assets** is the strongest predictor (feature importance: 0.35)
- ROE and Net Income also significant
- Market classification (KOSPI/KOSDAQ) has moderate impact

### 3. Model Performance
- **XGBoost** achieves highest accuracy (87.1%)
- All models exceed 84% accuracy
- Feature importance consistent across models

### 4. Value Identification
- ML models successfully identify undervalued firms
- Behavior patterns differ between undervalued and fairly valued firms

---

## Customization

### Change Number of Clusters

```python
# Cell 24: Modify k value
kmeans = KMeans(n_clusters=4, random_state=0)  # Try k=4
```

### Add More Features

```python
# Cell 18: Expand feature list
features = features + ['YOUR_NEW_FEATURE']
```

### Try Different Models

```python
# Cell 47: Add new models
from sklearn.svm import SVC
models['SVM'] = SVC(random_state=0)
```

---

## Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add comments for complex operations
- Update documentation for new features
- Test code before submitting PR

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{stockbuyback2023,
  title={Analysis of Stock Buyback Behavior in Korean Stock Market},
  author={Your Name and Co-authors},
  journal={Journal Name},
  year={2023},
  publisher={Publisher Name}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2023 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## Acknowledgments

- **Data Source**: TS2000 (KRX)
- **Libraries**: scikit-learn, XGBoost, pandas, matplotlib
- **Inspiration**: Research on corporate financial behavior
- **Contributors**: [Kim Jin Soo, Bae Hwan Suk]

---

## Contact

- **Author**: [Kim Jin Soo]
- **Email**: wlstn010203@khu.ac.kr
- **Institution**: [Kyoung-Hee university/Smart Lab]
- **Paper**: [(http://dx.doi.org/10.13088/jiis.2024.30.1.037)]


---

**Last Updated**: 2026-01-07 | **Version**: 1.0.0 | **Status**: Production Ready 
