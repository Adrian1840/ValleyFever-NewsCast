# News Media Coverage Predictors of Valley Fever Case Rates

## Project Overview

This project investigates whether news media coverage contributes to Valley
Fever case reporting patterns in high-incidence regions such as Fresno and Kern County, located in California's Central Valley. We combine environmental predictors from external sources and scraped Google News articles mentioning terms related to Valley Fever. News media features include article frequency, Central Valley county mentions, and risk-related terminology within these Google News articles. These features are evaluated using regression, Long Short-Term Memory (LSTM), and SARIMAX forecasting models. The goal is to evaluate how severity of the disease is
communicated to the public and whether media urgency corresponds to changes in reported case rates across Fresno and Kern County.

## Repository Contents

- `notebooks/01_news_scraping.ipynb`: Scrapes Google News articles.
- `notebooks/02_article_processing_nlp.ipynb`: Cleans article text, and extracts media-related features.
- `notebooks/03_preprocessing_lags.ipynb`: Merges news, environmental, and case-rate data; creates lagged variables and final model dataframe.
- `notebooks/04_regression_analysis.ipynb`: Runs regression models and influence diagnostics.
- `notebooks/05_forecasting_lstm.ipynb`: Runs LSTM, PFI, and forecasting evaluation.
- `notebooks/06_forecasting_sarimax.ipynb`: Runs SARIMAX and forecasting evaluation.

- `data/raw/`: Original Google News RSS outputs prior to article extraction.
- `data/external/`: Environmental and Valley Fever case-rate datasets adapted from the ValleyCast repository.
- `data/interim/`: Cleaned article-level datasets and intermediate monthly news aggregations.
- `data/processed/`: Final modeling datasets used for regression, LSTM, and SARIMAX forecasting analyses.
-  `src/`: Python modules for LSTM/SARIMAX helper functions, evaluation functions, and plotting functions.
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
│   ├── external/
│   │   ├── Fresno_Aggregate.csv
│   │   └── Kern_Aggregate.csv
│   │
│   ├── raw/
│   │   ├── google_news_rss_raw.csv
│   │   ├── google_news_rss_with_text.csv
│   │   └── article_counts_only.csv
│   │
│   ├── interim/
│   │   ├── google_news_rss_clean.csv
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
│   └── 05_forecasting_lstm.ipynb
│   └── 06_forecasting_sarimax.ipynb
│
├── src/
│   ├── lstm_model.py
│   ├── sarimax_model.py
│   ├── evaluation.py
│   └── plotting.py
│
├── results/
│   │
│   ├── regression/
│   │   ├── cooks_model.csv
│   │   └── dffits_model.csv
│   │
│   ├── lstm/
│   │   ├── fresno_lstm.png
│   │   ├── kern_lstm.png
│   │   └── hyperparameter_search/
│   │       ├── fresno_grid_search_results.csv
│   │       └── kern_grid_search_results.csv
│   │
│   ├── pfi/
│   │   ├── fresno_pfi.png
│   │   ├── kern_pfi.png
│   │   ├── fresno_pfis_100.csv
│   │   └── kern_pfis_100.csv
│   │
│   ├── pacf/
│   │   ├── fresno_pacf.png
│   │   └── kern_pacf.png
│   │
│   └── sarimax/
│       ├── fresno_sarimax.png
│       ├── kern_sarimax.png
│       ├── fresno_news.png
│       └── kern_news.png
│
│
└── paper/
    └── News_Media_Coverage_Predictors_of_Valley_Fever_Case_Rates.pdf
```

## Data Sources

- ValleyCast environmental and case-rate data: https://github.com/MBanuelos/ValleyCast
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

Run the notebooks in numerical order from `01` to `06`.

## Conference Submission

This repository accompanies the paper:

*News Media Coverage Predictors of Valley Fever Case Rates*

submitted to the Iberoamerican Congress on Pattern Recognition (CIARP 2026).


## Author
Adrian Lopez

Spring 2026
