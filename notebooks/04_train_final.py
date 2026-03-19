import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, classification_report
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib
import os

print("Loading cleaned data...")
df = pd.read_csv("data/cleaned_data.csv")
print(f"Loaded {len(df)} rows")

# ═══════════════════════════════════════════════════════════════════
# THE KEY INSIGHT:
# We divide the world into a grid of cells (2° x 2° boxes).
# For each cell we calculate: how many earthquakes happened there?
# What was the average magnitude? Max magnitude?
# THESE patterns are learnable — seismic zones are stable over time.
# ═══════════════════════════════════════════════════════════════════

print("\nBuilding seismic zone features...")

# Round lat/lon to nearest 2 degrees — creates grid cells
df["lat_bin"] = (df["Latitude"]  / 2).round() * 2
df["lon_bin"] = (df["Longitude"] / 2).round() * 2

# For each grid cell, calculate historical seismicity stats
zone_stats = df.groupby(["lat_bin", "lon_bin"]).agg(
    zone_count   = ("Magnitude", "count"),   # how many quakes here?
    zone_avg_mag = ("Magnitude", "mean"),    # average magnitude here?
    zone_max_mag = ("Magnitude", "max"),     # biggest quake here?
    zone_std_mag = ("Magnitude", "std"),     # how variable is it?
).reset_index()
zone_stats["zone_std_mag"] = zone_stats["zone_std_mag"].fillna(0)

# Merge these zone stats back into main dataframe
df = df.merge(zone_stats, on=["lat_bin", "lon_bin"], how="left")

print(f"Added zone features. Sample zone counts:")
print(df["zone_count"].describe().round(1))

# ── Define Seismic Hazard Level (our new target) ──────────────────
# This is based on BOTH the zone activity AND the earthquake depth
# High hazard = active zone + shallow depth (most dangerous combo)
def hazard_level(row):
    zone_active = row["zone_count"] > 50
    zone_very_active = row["zone_count"] > 200
    shallow = row["Depth"] < 70
    high_avg = row["zone_avg_mag"] >= 6.0

    if zone_very_active and shallow and high_avg:
        return "very_high"
    elif zone_active and (shallow or high_avg):
        return "high"
    elif zone_active:
        return "moderate"
    else:
        return "low"

df["hazard_level"] = df.apply(hazard_level, axis=1)

print(f"\nHazard level distribution:")
print(df["hazard_level"].value_counts())

# ── Features for training ─────────────────────────────────────────
features = [
    "Latitude", "Longitude", "Depth",
    "lat_sin", "lat_cos", "lon_sin", "lon_cos",
    "zone_count", "zone_avg_mag", "zone_max_mag", "zone_std_mag"
]

X = df[features]

# ═══════════════════════════════════════════════════════════════════
# MODEL 1 — Magnitude Regressor (now with zone features)
# ═══════════════════════════════════════════════════════════════════
print("\n── Model 1: Magnitude Regressor ──")

y_mag = df["Magnitude"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y_mag, test_size=0.2, random_state=42
)

mag_model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)
mag_model.fit(X_train, y_train, verbose=False)

y_pred = mag_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
print(f"  MAE : {mae:.3f}  (off by this much magnitude on average)")
print(f"  R²  : {r2:.3f}  (was 0.03 before — should be much better now!)")

print("\n  Sample predictions vs actual:")
print("  Predicted | Actual | Difference")
print("  ─────────────────────────────")
for pred, actual in list(zip(y_pred[:8], y_test.values[:8])):
    diff = abs(pred - actual)
    print(f"  {pred:.2f}      | {actual:.2f}   | {diff:.2f}")

# ═══════════════════════════════════════════════════════════════════
# MODEL 2 — Hazard Level Classifier
# ═══════════════════════════════════════════════════════════════════
print("\n── Model 2: Hazard Level Classifier ──")

le = LabelEncoder()
y_hazard = le.fit_transform(df["hazard_level"])
print(f"  Classes: {list(le.classes_)}")

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X, y_hazard, test_size=0.2, random_state=42, stratify=y_hazard
)

# Class weights to handle imbalance
counts = np.bincount(y_train2)
weights = len(y_train2) / (len(counts) * counts)
sample_weights = np.array([weights[y] for y in y_train2])

sev_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="mlogloss",
    random_state=42,
    verbosity=0
)
sev_model.fit(X_train2, y_train2, sample_weight=sample_weights)

y_pred2 = sev_model.predict(X_test2)
print("\n  Classification Report:")
print(classification_report(
    y_test2, y_pred2,
    target_names=le.classes_,
    zero_division=0
))

# ── Feature Importance ────────────────────────────────────────────
print("── What the model learned (feature importance) ──")
importances = mag_model.feature_importances_
for feat, imp in sorted(zip(features, importances),
                        key=lambda x: x[1], reverse=True):
    bar = "█" * int(imp * 60)
    print(f"  {feat:20s} {bar} {imp:.3f}")

# ── Save cleaned data with zone features for the app ─────────────
df.to_csv("data/final_data.csv", index=False)

# ── Save all models ───────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
joblib.dump(mag_model, "models/magnitude_model.pkl")
joblib.dump(sev_model, "models/severity_model.pkl")
joblib.dump(le,        "models/label_encoder.pkl")
joblib.dump(features,  "models/feature_names.pkl")
joblib.dump(zone_stats,"models/zone_stats.pkl")

print("\nAll models + zone stats saved!")
print("Done! ✓")

