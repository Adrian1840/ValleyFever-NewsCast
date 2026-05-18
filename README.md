# News Media Coverage Predictors of Valley Fever Case Rates

## Project Overview
This project looks at how affordable housing availability for low-income households has changed across U.S. states from 2014 to 2023. It also looks at whether housing shortages are associated with higher poverty exposure and student homelessness. Using state-level averages that were computed from county-level mobility metrics provided by the Urban Institute, I create vizualtions to highlight states with the most servere shortages and linear regression models that test these relationships. Ultimately, the results from this project can be used to inform which states should allocate resources for affordable housing and homelessness-prevention for students in public-schools across the nation.

## Project Structure
```text
VF-NewsMedia-Forecasting/
│
├── README.md
├── requirements.txt
├── environment.yml
│
├── data/
│   ├── raw/
│   │   ├── vf_case_rates/
│   │   ├── news_rss/
│   │   ├── article_text/
│   │   └── environmental/
│   │
│   ├── processed/
│   │   ├── news_monthly.csv
│   │   ├── news_nlp_monthly.csv
│   │   ├── vf_news_merged.csv
│   │   ├── vf_news_lagged.csv
│   │   └── lstm_ready_data.csv
│   │
│   └── final/
│       ├── regression_dataset.csv
│       └── lstm_dataset.csv
│
├── notebooks/
│   ├── 01_pygooglenews_collection.ipynb
│   ├── 02_newspaper4k_article_extraction.ipynb
│   ├── 03_nlp_feature_engineering.ipynb
│   ├── 04_vf_case_rate_preprocessing.ipynb
│   ├── 05_merge_news_vf_create_lags.ipynb
│   ├── 06_exploratory_plots.ipynb
│   ├── 07_regression_models.ipynb
│   ├── 08_influence_diagnostics_robustness.ipynb
│   ├── 09_lstm_preprocessing.ipynb
│   ├── 10_lstm_models.ipynb
│   ├── 11_lstm_permutation_importance.ipynb
│   └── 12_final_figures_for_poster.ipynb
│
├── figures/
│   ├── exploratory/
│   ├── regression/
│   ├── lstm/
│   └── poster/
│
├── results/
│   ├── regression_tables/
│   ├── influence_diagnostics/
│   ├── lstm_metrics/
│   └── model_comparisons/
│
└── src/
    ├── news_utils.py
    ├── nlp_utils.py
    ├── preprocessing_utils.py
    ├── regression_utils.py
    ├── lstm_utils.py
    └── plotting_utils.py
```

## Requirements
- Python 3.7+
- pandas
- matplotlib
- numpy
- jupyter (for local execution)

## Installation and Setup
### Local Execution
1. Clone this repository:
```text
git clone https://github.com/Adrian1840/MATH120_Final_Project/tree/main
cd python_final_project
```
2. Install required packages (if needed):
```text
pip install pandas matplotlib numpy jupyter
```
4. Launch Jupyter Notebook:
```text
jupyter notebook MATH120_Final_Project.ipynb
```

### Google Colab Execution
1. Open Google Colab
2. Upload the `...` file or connect to your GitHub repository
3. Run the first cell to automatically set up the environment

## Data Description
- **County-level mobility metrics** across 2014–2023: Contains variables on housing affordadability/availability, share of poverty exposure, and share of homeless public-school students.
   
## Analysis Features
- Data loading and cleaning
- Data wrangling with real longitudinal indicators
- State-by-year aggregation
- Clear trend visualization using Plotly line charts
- Exploratory association testing using scatterplots with fitted regression lines and simple linear regression (OLS).

## Key Learning Objectives Demonstrated
- File I/O with pandas in a reproducible project structure.
- Working with longitudinal county-level and state-level mobility data across multiple years.
- Data grouping and aggregation for state-year summaries.
- Exploratory data visualization with Plotly to analyze trends over time and compare states.
- Identification and comparison of states with systematically low affordable housing availability.
- Application of simple linear regression (OLS) to examine associations between affordable housing, poverty exposure, and student homelessness.
- Interpretation of regression outputs in order to inform policy on allocating resources for affordable housing and student homelessness prevention.

## Usage
Run all cells in `...` sequentially. The notebook will:

1. Set up the environment automatically
2. Load and clean the raw data
3. Perform statistical analysis
4. Generate visualizations
5. Save processed data to the `data/` folder

## Author
Adrian Lopez
MATH 120 - Fall/2025
