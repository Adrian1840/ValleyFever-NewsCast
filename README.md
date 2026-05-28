# News Media Coverage Predictors of Valley Fever Case Rates

## Project Overview

This project investigates whether public awareness communicated through news media coverage contributes to Valley
Fever case reporting patterns in high-incidence regions of California's Central Valley such as Fresno and Kern County. We combine environmental predictors from other sources and news media features scraped from Google News articles mentioning terms related to Valley Fever. News media features include article frequency, Central Valley county mentions, and risk-related terminology within these Google News articles. These features are evaluated using regression, Long Short-Term Memory (LSTM), and SARIMAX forecasting models. The goal is to evaluate whether media urgency and news coverage corresponds to changes in reported case rates across Fresno and Kern County.

## Repository Contents

- `notebooks/01_news_scraping.ipynb`: Scrapes Google News articles.
- `notebooks/02_article_processing_nlp.ipynb`: Cleans article text, and extracts media-related features.
- `notebooks/03_preprocessing_lags.ipynb`: Merges news, environmental, and case-rate data; creates lagged variables.
- `notebooks/04_regression_analysis.ipynb`: Runs regression models and influence diagnostics.
- `notebooks/05_forecasting_lstm_sarimax.ipynb`: Runs LSTM, PFI, SARIMAX, and forecasting evaluation.
- `data/raw/`: Original Google News RSS outputs prior to article extraction.
- `data/external/`: Environmental and Valley Fever case-rate datasets adapted from the ValleyCast repository.
- `data/interim/`: Cleaned article-level datasets and intermediate monthly news aggregations.
- `data/processed/`: Final modeling datasets used for regression, LSTM, and SARIMAX forecasting analyses.
-  `results/`: Generated figures, forecasting outputs, hyperparameter search results, and evaluation tables.
-  `paper/`: Final conference paper manuscript.


## Project Structure
```text
ValleyFever-NewsCast/
│
├── README.md
├── requirements.txt
│
├── data/
│   │
│   ├── raw/
│   │   └── google_news_rss_raw.csv
│   │
│   ├── external/
│   │   ├── Fresno_Aggregate.csv
│   │   └── Kern_Aggregate.csv
│   │
│   ├── interim/
│   │   ├── google_news_rss_clean.csv
│   │   └── monthly_news_counts.csv
│   │
│   └── processed/
│       ├── final_model_dataframe.csv
│       ├── news_features_lag3.csv
│       └── news_features_unlagged.csv
│
├── notebooks/
│   ├── 01_news_scraping.ipynb
│   ├── 02_article_processing_nlp.ipynb
│   ├── 03_time_lag_preprocessing.ipynb
│   ├── 04_regression_analysis.ipynb
│   └── 05_forecasting_lstm_sarimax.ipynb
│
├── results/
│   ├── figures/
│   ├── forecasts/
│   ├── hyperparameter_search/
│       ├── fresno_grid_search_results.csv
│       └── kern_grid_search_results.csv
|   └── pfi/
│       ├── fresno_pfis_100.csv
│       └── kern_pfis_100.csv
│
└── paper/
    └── News_Media_Coverage_Predictors_of_Valley_Fever_Case_Rates.pdf
```

## Data Sources

- ValleyCast environmental and case-rate data
- Google News RSS results extracted using `pygooglenews` 
- Newspaper article text extracted using `newspaper4k`

## Main Methods

- Natural Language Processing
- Lagged media feature engineering
- Regression with influence diagnostics
- LSTM time-series forecasting
- SARIMAX forecasting
- Permutation Feature Importance

## Reproducibility

Run the notebooks in numerical order from `01` to `04`.

## Author
Adrian Lopez
MATH 120 - Fall/2025
