# News Media Coverage Predictors of Valley Fever Case Rates

## Project Overview

This project investigates whether public awareness communicated through news media contributes to observed Valley
Fever case reporting patterns in high-incidence regions of California's Central Valley such as Fresno and Kern County. We combine environmental predictors from other sources and news media features scraped from Google News articles mentioning terms related to Valley Fever. News media features include article frequency, Central Valley county mentions, and risk-related terminology within these Google News articles. These features are evaluated using regression, Long Short-Term Memory (LSTM), and SARIMAX forecasting models. The goal is to evaluate whether media urgency and news coverage corresponds to changes in reported case rates across Fresno and Kern County.

## Repository Contents

- `notebooks/01_news_scraping_nlp.ipynb`: Scrapes Google News articles and extracts media-related features.
- `notebooks/02_preprocessing_lags.ipynb`: Merges news, environmental, and case-rate data; creates lagged variables.
- `notebooks/03_regression_analysis.ipynb`: Runs regression models and influence diagnostics.
- `notebooks/04_forecasting_lstm_sarimax.ipynb`: Runs LSTM, PFI, SARIMAX, and forecasting evaluation.
- `data/`: Raw, processed, and external datasets.
- `results/`: Figures, tables, and model outputs.
- `paper/`: Final research paper.

  
## Project Structure
```text
VF-NewsMedia-Forecasting/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── notebooks/
│   ├── 01_news_scraping_nlp.ipynb
│   ├── 02_preprocessing_lags.ipynb
│   ├── 03_regression_analysis.ipynb
│   └── 04_forecasting_lstm_sarimax.ipynb
│
├── results/
│   ├── figures/
│   ├── tables/
│   └── model_outputs/
│
└── paper/
    └── News_Media_Coverage_Predictors_of_Valley_Fever_Case_Rates.pdf
```

## Data Sources

- ValleyCast environmental and case-rate data
- Google News RSS results
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
