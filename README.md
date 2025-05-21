# YSDA RecSys 2025 Lavka Solution

This repository contains a solution for the [Kaggle competition YSDA RecSys 2025 Lavka](https://www.kaggle.com/competitions/ysda-recsys-2025-lavka). The goal is to predict user-item interactions for a grocery recommendation system.

## Getting started

```bash
# Download competition data
kaggle competitions download -c ysda-recsys-2025-lavka
unzip ysda-recsys-2025-lavka.zip

# Install dependencies
pip install -r requirements.txt

# Prepare daily features
python precalc_features.py

# Run training
python trainer.py

# Optimizes model params automatically
python optimize.py
```

## Key Components

### 1. **`calculate_stats.py`**
- Generates collaborative and statistical features:
  - Collaborative filtering: NPMI, Jaccard similarity, SVD embeddings
  - Log1p-counts for actions (views, purchases) grouped by user/product/store/city
  - Click-through rate (CTR) features
  - Temporal features (hour, weekday, time_of_day, etc.)

### 2. **`Featurizer`**
- Computes features in **causal mode** (no future data leakage)
- Stores daily feature snapshots for efficient joins
- Uses sliding windows (3/7/30/all days) for time-decay modeling

### 3. **`Preprocessor`**
- Handles competition-specific rules:
  - Makes train/val split
  - Filters small request_ids

### 4. **`Trainer`**
- Supports three training modes:
  - **Random lag**: Random historical offset (1-30 days) for the each sample
  - **Constant lag**: Fixed historical offset (e.g., 7 days)
  - **Ensemble**: Combines random + all constant lags
- Uses **CatBoost**:
  - `CatBoostClassifier` for binary classification (view/purchase)
  - `CatBoostRanker` with pairwise NDCG optimization
- Implements smart caching for fast iteration

### 5. **Hyperparameter Optimization**
- **`optimize.py`**: Uses Optuna for GPU-accelerated parameter tuning

### 6. **Metrics & Submission**
- Calculates NDCG@10 for ranking evaluation
- Handles ensemble result merging
- Produces Kaggle-ready CSV submissions

## Key Implementation Details

### 1. **Causal Feature Engineering**
- All features are computed per day with no future data leakage
- Windows handle time series changes

### 2. **Ensemble Strategy**
- Combines 31 models (random lag + 30 constant lags)
- Uses day-specific weighting for constant lags

### 3. **Efficiency Features**
- Parquet caching for fast feature reuse
- GPU acceleration for CatBoost and SVD
