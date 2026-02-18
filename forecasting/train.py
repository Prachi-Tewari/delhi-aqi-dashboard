"""
AQI Forecasting — Training Pipeline
=====================================
Conference-grade training with:
  1. Pollution regime clustering (KMeans on diurnal/seasonal patterns)
  2. Global + regime-specific LightGBM models
  3. XGBoost baseline
  4. Persistence / ARIMA baselines
  5. SHAP interpretability
  6. 6-hour recursive multi-step evaluation

Usage:
    python -m forecasting.train --data /path/to/Delhi_AQI_Combined_2020_2024.csv
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from forecasting.features import (
    load_and_clean, handle_missing, cap_outliers,
    build_features, get_feature_columns, time_split,
    POLLUTANTS, METEO_FEATURES, LAG_HOURS, ROLLING_WINDOWS,
)

MODEL_DIR = PROJECT_ROOT / "forecasting" / "models"
MODEL_DIR.mkdir(exist_ok=True)

HORIZON = 6  # hours


# ═════════════════════════════════════════════════════════════════════
# METRICS
# ═════════════════════════════════════════════════════════════════════
def eval_metrics(y_true, y_pred, prefix=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    # Spike performance: MAE for top-10% AQI events
    threshold = np.percentile(y_true, 90)
    spike_mask = y_true >= threshold
    spike_mae = mean_absolute_error(y_true[spike_mask], y_pred[spike_mask]) if spike_mask.sum() > 10 else np.nan
    return {
        f"{prefix}MAE": round(mae, 2),
        f"{prefix}RMSE": round(rmse, 2),
        f"{prefix}R2": round(r2, 4),
        f"{prefix}Spike_MAE": round(spike_mae, 2) if not np.isnan(spike_mae) else "N/A",
    }


# ═════════════════════════════════════════════════════════════════════
# BASELINES
# ═════════════════════════════════════════════════════════════════════
def persistence_baseline(test_df, target="aqi"):
    """Naive persistence: predict AQI(t) = AQI(t-1)."""
    y_true = test_df[target].values
    y_pred = test_df[f"{target}_lag1"].values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    return eval_metrics(y_true[mask], y_pred[mask], prefix="Persist_")


def arima_baseline(train_df, test_df, target="aqi"):
    """Simple ARIMA(2,1,2) baseline on hourly AQI."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
        # Use last 2 weeks of training for speed
        train_series = train_df[target].dropna().iloc[-336:]
        model = ARIMA(train_series, order=(2, 1, 2))
        fit = model.fit()
        preds = fit.forecast(steps=min(len(test_df), 720))  # max 30 days
        y_true = test_df[target].dropna().iloc[:len(preds)]
        return eval_metrics(y_true.values, preds.values[:len(y_true)], prefix="ARIMA_")
    except Exception as e:
        print(f"  ARIMA baseline failed: {e}")
        return {"ARIMA_MAE": "N/A", "ARIMA_RMSE": "N/A", "ARIMA_R2": "N/A"}


# ═════════════════════════════════════════════════════════════════════
# REGIME CLUSTERING
# ═════════════════════════════════════════════════════════════════════
def build_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build daily-aggregated features for regime clustering."""
    daily = df.resample("1D").agg({
        "aqi": ["mean", "std", "max", "min"],
        "pm25": "mean",
        "pm10": "mean",
        "temperature": "mean",
        "humidity": "mean",
        "wind_speed": "mean",
    })
    daily.columns = ["_".join(c).strip("_") for c in daily.columns]
    daily["aqi_range"] = daily["aqi_max"] - daily["aqi_min"]
    daily = daily.dropna()
    return daily


def cluster_regimes(train_df: pd.DataFrame, n_clusters: int = 3):
    """
    KMeans clustering on daily pollution profiles.
    Returns fitted scaler, kmeans model, and regime labels for each hour.
    """
    daily = build_regime_features(train_df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(daily)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(X_scaled)
    daily["regime"] = km.labels_
    return scaler, km, daily[["regime"]]


def assign_regimes(df: pd.DataFrame, scaler, km_model) -> pd.Series:
    """Assign regime labels to each hour using daily aggregation."""
    daily = build_regime_features(df)
    if len(daily) == 0:
        return pd.Series(0, index=df.index, name="regime")
    X_scaled = scaler.transform(daily)
    daily["regime"] = km_model.predict(X_scaled)
    # Map daily regime to hourly
    regime_hourly = daily[["regime"]].reindex(df.index, method="ffill")
    regime_hourly["regime"] = regime_hourly["regime"].fillna(0).astype(int)
    return regime_hourly["regime"]


# ═════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ═════════════════════════════════════════════════════════════════════
def train_lightgbm(X_train, y_train, X_val, y_val, params=None):
    """Train LightGBM with early stopping."""
    import lightgbm as lgb

    default_params = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "num_leaves": 127,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
        "n_estimators": 2000,
        "random_state": 42,
    }
    if params:
        default_params.update(params)

    model = lgb.LGBMRegressor(**default_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost with early stopping."""
    import xgboost as xgb

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=2000,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        early_stopping_rounds=50,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


# ═════════════════════════════════════════════════════════════════════
# RECURSIVE 6-HOUR FORECASTING
# ═════════════════════════════════════════════════════════════════════
def recursive_forecast(model, df: pd.DataFrame, feat_cols: list,
                       target: str = "aqi", horizon: int = HORIZON) -> list:
    """
    Recursive multi-step forecast from the last available row.
    Returns list of predicted AQI values for +1h to +{horizon}h.
    """
    predictions = []
    # Work on a copy
    work = df.copy()
    last_idx = work.index[-1]

    for step in range(1, horizon + 1):
        # Use the last available row's features to predict
        X_input = work.loc[[last_idx], feat_cols]
        if X_input.isna().any(axis=1).values[0]:
            X_input = X_input.fillna(method="ffill", axis=1).fillna(0)

        pred = float(model.predict(X_input)[0])
        pred = max(pred, 0)  # AQI can't be negative
        predictions.append(pred)

        # Create next timestamp row and update lag features
        next_ts = last_idx + pd.Timedelta(hours=1)
        new_row = work.loc[[last_idx]].copy()
        new_row.index = [next_ts]

        # Shift AQI target and lags
        new_row[target] = pred
        for lag in LAG_HOURS:
            col = f"{target}_lag{lag}"
            if col in new_row.columns:
                if lag == 1:
                    new_row[col] = work.loc[last_idx, target]
                else:
                    # Look back `lag` rows
                    look_ts = next_ts - pd.Timedelta(hours=lag)
                    if look_ts in work.index:
                        new_row[col] = work.loc[look_ts, target]

        # Update rolling features
        for w in ROLLING_WINDOWS:
            rmean_col = f"{target}_rmean{w}"
            rstd_col = f"{target}_rstd{w}"
            if rmean_col in new_row.columns:
                recent = work[target].iloc[-w:]
                new_row[rmean_col] = recent.mean()
                new_row[rstd_col] = recent.std() if len(recent) > 1 else 0

        # Update rate-of-change
        if f"{target}_delta1" in new_row.columns:
            new_row[f"{target}_delta1"] = pred - work.loc[last_idx, target]

        # Update temporal features
        new_row["hour"] = next_ts.hour
        new_row["hour_sin"] = np.sin(2 * np.pi * next_ts.hour / 24)
        new_row["hour_cos"] = np.cos(2 * np.pi * next_ts.hour / 24)
        new_row["dow"] = next_ts.dayofweek
        new_row["dow_sin"] = np.sin(2 * np.pi * next_ts.dayofweek / 7)
        new_row["dow_cos"] = np.cos(2 * np.pi * next_ts.dayofweek / 7)
        new_row["month"] = next_ts.month
        new_row["month_sin"] = np.sin(2 * np.pi * next_ts.month / 12)
        new_row["month_cos"] = np.cos(2 * np.pi * next_ts.month / 12)
        new_row["is_weekend"] = int(next_ts.dayofweek >= 5)

        work = pd.concat([work, new_row])
        last_idx = next_ts

    return predictions


def evaluate_recursive(model, df: pd.DataFrame, feat_cols: list,
                       target: str = "aqi", horizon: int = HORIZON,
                       n_eval_points: int = 200) -> dict:
    """
    Evaluate recursive forecast at multiple points in the dataset.
    Returns per-horizon metrics.
    """
    results = {h: {"true": [], "pred": []} for h in range(1, horizon + 1)}
    valid_indices = df.dropna(subset=feat_cols + [target]).index

    if len(valid_indices) < horizon + 168 + n_eval_points:
        n_eval_points = max(50, len(valid_indices) - horizon - 168)

    # Sample evaluation points evenly from available range
    eval_start = 168  # need at least 168 lags
    eval_end = len(valid_indices) - horizon
    if eval_end <= eval_start:
        return {}

    step_size = max(1, (eval_end - eval_start) // n_eval_points)
    eval_positions = range(eval_start, eval_end, step_size)

    for pos in eval_positions:
        # Use data up to this point
        history = df.iloc[:pos + 1]
        preds = recursive_forecast(model, history, feat_cols, target, horizon)

        for h in range(1, horizon + 1):
            future_pos = pos + h
            if future_pos < len(df):
                true_val = df.iloc[future_pos][target]
                if not np.isnan(true_val):
                    results[h]["true"].append(true_val)
                    results[h]["pred"].append(preds[h - 1])

    metrics = {}
    for h in range(1, horizon + 1):
        if len(results[h]["true"]) > 0:
            y_t = np.array(results[h]["true"])
            y_p = np.array(results[h]["pred"])
            metrics[f"+{h}h"] = eval_metrics(y_t, y_p)

    return metrics


# ═════════════════════════════════════════════════════════════════════
# SHAP ANALYSIS
# ═════════════════════════════════════════════════════════════════════
def compute_shap(model, X_sample, feat_cols, output_dir):
    """Compute and save SHAP values and importance."""
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # Summary bar plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feat_cols,
                          plot_type="bar", show=False, max_display=25)
        plt.tight_layout()
        plt.savefig(output_dir / "shap_importance.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Beeswarm
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feat_cols,
                          show=False, max_display=25)
        plt.tight_layout()
        plt.savefig(output_dir / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Save mean absolute SHAP values as JSON
        mean_shap = np.abs(shap_values).mean(axis=0)
        importance = dict(sorted(
            zip(feat_cols, mean_shap.tolist()),
            key=lambda x: x[1], reverse=True
        ))
        with open(output_dir / "shap_importance.json", "w") as f:
            json.dump(importance, f, indent=2)

        print(f"  SHAP analysis saved to {output_dir}")
        return importance
    except Exception as e:
        print(f"  SHAP analysis failed: {e}")
        return {}


# ═════════════════════════════════════════════════════════════════════
# MAIN TRAINING PIPELINE
# ═════════════════════════════════════════════════════════════════════
def main(data_path: str):
    print("=" * 70)
    print("AQI FORECASTING — TRAINING PIPELINE")
    print("=" * 70)

    # ── 1. DATA PREPARATION ────────────────────────────────────────
    print("\n[1/7] Loading and cleaning data ...")
    df = load_and_clean(data_path)
    print(f"  Shape after cleaning: {df.shape}")
    print(f"  Date range: {df.index.min()} → {df.index.max()}")
    print(f"  AQI stats: mean={df['aqi'].mean():.1f}, "
          f"median={df['aqi'].median():.1f}, "
          f"max={df['aqi'].max():.1f}")

    # Missing data
    df = handle_missing(df)
    aqi_missing = df["aqi"].isna().sum()
    print(f"  AQI missing after interpolation: {aqi_missing} "
          f"({100*aqi_missing/len(df):.1f}%)")

    # Outlier capping
    df = cap_outliers(df, "aqi")

    # ── 2. FEATURE ENGINEERING ─────────────────────────────────────
    print("\n[2/7] Feature engineering ...")
    df = build_features(df, target="aqi", include_meteo=True)
    feat_cols = get_feature_columns(df, target="aqi")
    print(f"  Total features: {len(feat_cols)}")

    # Drop rows with NaN in features or target
    before = len(df)
    df_clean = df.dropna(subset=feat_cols + ["aqi"])
    print(f"  Rows after dropping NaN: {len(df_clean)} (dropped {before - len(df_clean)})")

    # ── 3. TIME-BASED SPLIT ────────────────────────────────────────
    print("\n[3/7] Time-based splitting ...")
    train, val, test = time_split(df_clean)
    print(f"  Train: {len(train)} rows ({train.index.min()} → {train.index.max()})")
    print(f"  Val:   {len(val)} rows ({val.index.min()} → {val.index.max()})")
    print(f"  Test:  {len(test)} rows ({test.index.min()} → {test.index.max()})")

    X_train, y_train = train[feat_cols], train["aqi"]
    X_val, y_val = val[feat_cols], val["aqi"]
    X_test, y_test = test[feat_cols], test["aqi"]

    # ── 4. REGIME CLUSTERING ──────────────────────────────────────
    print("\n[4/7] Pollution regime clustering ...")
    regime_scaler, regime_km, train_daily_regimes = cluster_regimes(train, n_clusters=3)

    # Assign regimes to all splits
    train["regime"] = assign_regimes(train, regime_scaler, regime_km)
    val["regime"] = assign_regimes(val, regime_scaler, regime_km)
    test["regime"] = assign_regimes(test, regime_scaler, regime_km)

    regime_counts = train["regime"].value_counts().sort_index()
    regime_labels = {
        int(r): f"Regime {r} (n={c})" for r, c in regime_counts.items()
    }
    print(f"  Regime distribution (train):")
    for r, label in regime_labels.items():
        mean_aqi = train.loc[train["regime"] == r, "aqi"].mean()
        print(f"    {label}: mean AQI = {mean_aqi:.1f}")

    # ── 5. MODEL TRAINING ─────────────────────────────────────────
    print("\n[5/7] Training models ...")

    all_results = {}

    # -- A) Persistence baseline
    print("  [A] Persistence baseline ...")
    all_results["Persistence"] = persistence_baseline(test, "aqi")
    print(f"      {all_results['Persistence']}")

    # -- B) Global LightGBM
    print("  [B] Global LightGBM ...")
    lgbm_global = train_lightgbm(X_train, y_train, X_val, y_val)
    pred_test_lgbm = lgbm_global.predict(X_test)
    all_results["LightGBM_Global"] = eval_metrics(y_test.values, pred_test_lgbm, prefix="")
    print(f"      {all_results['LightGBM_Global']}")

    # -- C) Global XGBoost
    print("  [C] Global XGBoost ...")
    xgb_global = train_xgboost(X_train, y_train, X_val, y_val)
    pred_test_xgb = xgb_global.predict(X_test)
    all_results["XGBoost_Global"] = eval_metrics(y_test.values, pred_test_xgb, prefix="")
    print(f"      {all_results['XGBoost_Global']}")

    # -- D) Regime-specific LightGBM
    print("  [D] Regime-specific LightGBM ...")
    regime_models = {}
    pred_test_regime = np.zeros(len(test))

    for regime_id in sorted(train["regime"].unique()):
        tr_mask = train["regime"] == regime_id
        va_mask = val["regime"] == regime_id
        te_mask = test["regime"] == regime_id

        if tr_mask.sum() < 100 or va_mask.sum() < 20:
            print(f"    Regime {regime_id}: too few samples, using global model")
            regime_models[regime_id] = lgbm_global
            pred_test_regime[te_mask.values] = lgbm_global.predict(X_test[te_mask.values])
            continue

        X_tr_r = X_train[tr_mask.values]
        y_tr_r = y_train[tr_mask.values]
        X_va_r = X_val[va_mask.values]
        y_va_r = y_val[va_mask.values]

        rm = train_lightgbm(X_tr_r, y_tr_r, X_va_r, y_va_r)
        regime_models[regime_id] = rm
        if te_mask.sum() > 0:
            pred_test_regime[te_mask.values] = rm.predict(X_test[te_mask.values])
            r_metrics = eval_metrics(
                y_test.values[te_mask.values],
                pred_test_regime[te_mask.values]
            )
            print(f"    Regime {regime_id}: {r_metrics}")

    all_results["LightGBM_Regime"] = eval_metrics(y_test.values, pred_test_regime)
    print(f"      Combined: {all_results['LightGBM_Regime']}")

    # ── 6. SELECT BEST MODEL ──────────────────────────────────────
    print("\n[6/7] Model comparison & selection ...")

    # Determine best by MAE
    model_map = {
        "LightGBM_Global": lgbm_global,
        "XGBoost_Global": xgb_global,
    }
    best_name = min(
        ["LightGBM_Global", "XGBoost_Global", "LightGBM_Regime"],
        key=lambda k: all_results[k]["MAE"]
    )
    print(f"\n  Best model (1-step MAE): {best_name}")
    print(f"  Metrics: {all_results[best_name]}")

    # Choose the actual model object
    if best_name == "LightGBM_Regime":
        best_model = lgbm_global  # use global as primary; regime models saved separately
        use_regime = True
    else:
        best_model = model_map[best_name]
        use_regime = False

    # ── 6b. Recursive multi-step evaluation on test set ────────── 
    print("\n  Evaluating 6-hour recursive forecast ...")
    recursive_metrics = evaluate_recursive(best_model, test, feat_cols,
                                           "aqi", HORIZON, n_eval_points=150)
    for h, m in recursive_metrics.items():
        print(f"    {h}: {m}")

    # ── 7. SAVE ARTIFACTS ─────────────────────────────────────────
    print("\n[7/7] Saving model artifacts ...")

    # Save best model
    joblib.dump(best_model, MODEL_DIR / "best_model.pkl")
    print(f"  Saved best_model.pkl ({best_name})")

    # Save regime clustering artifacts
    joblib.dump(regime_scaler, MODEL_DIR / "regime_scaler.pkl")
    joblib.dump(regime_km, MODEL_DIR / "regime_kmeans.pkl")
    joblib.dump(regime_models, MODEL_DIR / "regime_models.pkl")
    print("  Saved regime clustering artifacts")

    # Save feature column list
    with open(MODEL_DIR / "feature_cols.json", "w") as f:
        json.dump(feat_cols, f)
    print(f"  Saved feature_cols.json ({len(feat_cols)} features)")

    # Save results
    comparison = {
        "model_comparison": all_results,
        "best_model": best_name,
        "use_regime": use_regime,
        "recursive_metrics": recursive_metrics,
        "regime_labels": regime_labels,
        "horizon": HORIZON,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "n_features": len(feat_cols),
    }
    with open(MODEL_DIR / "training_results.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    print("  Saved training_results.json")

    # SHAP analysis on sample
    print("\n  Running SHAP analysis ...")
    shap_sample = X_test.sample(min(500, len(X_test)), random_state=42)
    compute_shap(best_model, shap_sample, feat_cols, MODEL_DIR)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nModel comparison:")
    comparison_df = pd.DataFrame(all_results).T
    print(comparison_df.to_string())

    if recursive_metrics:
        print(f"\n6-hour recursive forecast performance:")
        rec_df = pd.DataFrame(recursive_metrics).T
        print(rec_df.to_string())

    return best_model, feat_cols, all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    args = parser.parse_args()
    main(args.data)
