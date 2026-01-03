# Exploratory Data Analysis (EDA) Repository

A comprehensive collection of Jupyter notebooks demonstrating various exploratory data analysis techniques, statistical methods, and machine learning algorithms using Python.

## 📋 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Topics Covered](#topics-covered)
- [Technologies Used](#technologies-used)
- [Notebooks Description](#notebooks-description)
- [Power BI Visualizations](#power-bi-visualizations)
- [Getting Started](#getting-started)
- [Requirements](#requirements)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This repository contains a collection of practical examples and exercises covering fundamental and advanced topics in exploratory data analysis, statistical analysis, and machine learning. Each notebook provides hands-on implementations with real-world datasets, making it an excellent resource for data science learners and practitioners.

## 📁 Repository Structure

```
Exploratory-Data-Analysis/
│
├── Are American people happy. ipynb
├── Basics of EDA.ipynb
├── Central Tendency and Similarity Measures.ipynb
├── Data preprocessing.ipynb
├── Detecting Outliers.ipynb
├── How about some correlation exercises.ipynb
├── Predicting Titanic Survivors with Data Analytics.ipynb
├── Predicting Titanic Survivors with Data Analytics - Report. docx
├── Regression and Multiple Regression.ipynb
├── So, Classification. ipynb
└── powerbi/
    ├── A little data viz.pdf
    ├── DataViz
    └── DataViz. pbix
```

## 📊 Topics Covered

### Statistical Analysis
- **Central Tendency Measures**: Mean, Median, Mode, Quartiles
- **Similarity Measures**: Correlation analysis and similarity metrics
- **Outlier Detection**:  IQR method, Z-score, statistical techniques

### Data Preprocessing
- **Data Cleaning**: Handling missing values, data validation
- **Data Transformation**: Encoding categorical variables, feature engineering
- **Data Quality Assessment**: Completeness analysis, rule-based validation

### Regression Analysis
- **Simple Linear Regression**: Single predictor models
- **Multiple Regression**: Multi-variable predictive modeling
- **Model Evaluation**: R-squared, F-statistics, coefficient interpretation

### Classification
- **Decision Trees**: Tree-based classification
- **Naive Bayes**: Probabilistic classification
- **Model Assessment**: Confusion matrices, accuracy scores, classification reports

### Association Rule Mining
- **Apriori Algorithm**: Frequent pattern mining
- **Association Rules**: Support, confidence, and lift metrics
- **Survival Analysis**: Titanic dataset analysis using association rules

### Data Visualization
- **Statistical Plots**: Scatter plots, distributions
- **Power BI Dashboards**: Interactive visualizations
- **Exploratory Visualizations**: Pattern identification and trend analysis

## 🛠️ Technologies Used

- **Python 3.x**
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-learn**:  Machine learning algorithms
- **Statsmodels**: Statistical modeling
- **MLxtend**: Advanced machine learning extensions
- **Apyori**: Association rule mining
- **Power BI**: Business intelligence and visualization

## 📚 Notebooks Description

### 1. Basics of EDA. ipynb
Introduction to exploratory data analysis using a job dataset with attributes like salary, education, and prestige. 
- Data loading and inspection
- Creating data subsets
- Basic pandas operations

### 2. Data preprocessing.ipynb
Comprehensive data cleaning and preprocessing techniques using an income dataset (1000 observations).
- Identifying missing values (88 gender, 109 income, 93 tax)
- Data validation with business rules
- Handling inconsistent categorical values
- Imputation strategies
- Data quality assessment (73. 3% complete observations)

### 3. Central Tendency and Similarity Measures. ipynb
Statistical analysis of salary and occupation data (45 job types).
- Calculating min, max, quartiles, mean, and median
- Distribution analysis across different attributes
- Comparative statistics for salary, education, and prestige

### 4. Detecting Outliers.ipynb
Outlier detection methods using the mtcars dataset (32 vehicles).
- Statistical outlier detection techniques
- IQR (Interquartile Range) method
- Visualization of outliers
- Impact assessment on analysis

### 5. How about some correlation exercises.ipynb
Correlation analysis and relationship exploration between variables.
- Pearson correlation
- Correlation matrices
- Identifying strong and weak relationships

### 6. Predicting Titanic Survivors with Data Analytics. ipynb
Association rule mining applied to the Titanic dataset (2,201 passengers).
- Data transformation for transaction format
- One-hot encoding
- Apriori algorithm implementation
- Extracting survival patterns: 
  - Female passengers in 1st class:  97.2% survival (Lift: 3.010)
  - Male passengers in crew: 99.6% non-survival correlation
  - Class-based survival patterns
  - Age and gender impact on survival

### 7. Regression and Multiple Regression.ipynb
Predictive modeling using album sales data (200 albums).
- **Simple Regression**: Sales vs.  Advertising Budget
  - R-squared: 0.331
  - F-statistic: 98.04 (p < 0.001)
  - Coefficient:  0.0955
- **Multiple Regression**: Sales vs. Advertising, Airplay, Attractiveness
  - R-squared:  0.617
  - F-statistic:  105.2 (p < 0.001)
  - Advertising coefficient: 0.0892
  - Airplay coefficient: 3.4485
- Sales prediction capabilities
- Scatter plot visualizations
- Model comparison and interpretation

### 8. So, Classification.ipynb
Binary classification models for income prediction (32,560 individuals).
- **Features**: Age, work type, education, occupation, marital status, race, gender, capital gain/loss, hours per week
- **Target**: Income (<=50K or >50K)
- Decision Tree Classifier
- Naive Bayes Classifier (GaussianNB)
- Label encoding for categorical variables
- Train-test split (80-20)
- Model evaluation metrics

### 9. Are American people happy.ipynb
Exploratory analysis of happiness indicators in the American population.
- Happiness metrics exploration
- Demographic analysis
- Statistical summaries

## 📊 Power BI Visualizations

The `powerbi/` directory contains: 
- **DataViz. pbix**: Power BI interactive dashboard file
- **A little data viz. pdf**: Exported visualization report
- Advanced data visualization examples
- Interactive filtering and drill-down capabilities

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.7+ installed on your system.

### Installation

1. Clone the repository:
```bash
git clone https://github.com/TheJegede/Exploratory-Data-Analysis.git
cd Exploratory-Data-Analysis
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels mlxtend apyori jupyter
```

## 📦 Requirements

```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
statsmodels>=0.12.0
mlxtend>=0.19.0
apyori>=1.1.2
jupyter>=1.0.0
```

## 💻 Usage

1. Launch Jupyter Notebook:
```bash
jupyter notebook
```

2. Navigate to the desired notebook and open it

3. Run cells sequentially to reproduce the analysis

4. Modify code and experiment with different parameters

### Example:  Running Regression Analysis

```python
import pandas as pd
import statsmodels.api as sm

# Load data
data = pd.read_csv("Lab Album Sales.csv")

# Prepare features
X = data[['AdvertBudget']]
y = data['totalsales']

# Fit model
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

# View results
print(model.summary())
```

## 📈 Key Insights and Findings

### Data Quality
- Importance of data validation (only 45% of income data passed all business rules)
- Impact of missing data on analysis completeness

### Statistical Patterns
- Strong correlations between education level and salary
- Occupational prestige varies significantly across job types

### Predictive Modeling
- Advertising budget and airplay are strong predictors of album sales
- Multiple regression improves prediction accuracy by 86% over simple regression

### Classification Performance
- Decision trees and Naive Bayes show different strengths for income prediction
- Feature encoding crucial for model performance

### Survival Analysis (Titanic)
- Gender and class were primary survival determinants
- Female passengers in 1st/2nd class had highest survival rates
- Male crew members had lowest survival rates

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.  For major changes: 

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Notes

- All datasets used in these notebooks are for educational purposes
- Some notebooks may require specific data files (CSV format)
- The repository demonstrates best practices in data analysis workflows
- Code is documented with markdown cells explaining each step

## 🎓 Learning Objectives

By working through these notebooks, you will: 
- Master fundamental EDA techniques
- Understand statistical measures and their applications
- Learn data preprocessing and cleaning strategies
- Build and evaluate regression models
- Implement classification algorithms
- Apply association rule mining
- Create effective data visualizations
- Develop end-to-end data analysis workflows

## 📧 Contact

**Author**: TheJegede  
**Repository**: [Exploratory-Data-Analysis](https://github.com/TheJegede/Exploratory-Data-Analysis)

## ⭐ Acknowledgments

- Dataset sources from various public domain repositories
- Inspiration from data science community best practices
- Statistical methods from classic literature

---

**Last Updated**: January 2026  
**Status**: Active Development

If you find this repository helpful, please consider giving it a ⭐! 
