import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, classification_report
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib
import os

print("Loading master dataset...")
df = pd.read_csv("data/master_dataset.csv", low_memory=False)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
print(f"Loaded {len(df)} rows | {df['Date'].min().date()} → {df['Date'].max().date()}")

# ═══════════════════════════════════════════════════════════════════
# TEMPORAL SPLIT — train on past, test on future
# No data leakage. This is how real models are evaluated.
# ═══════════════════════════════════════════════════════════════════
SPLIT_YEAR = 2015

train_df = df[df["Date"].dt.year <  SPLIT_YEAR].copy()
test_df  = df[df["Date"].dt.year >= SPLIT_YEAR].copy()

# Safety check — print what we actually have
print(f"\nDate values sample: {df['Date'].head(3).tolist()}")
print(f"Year range in data: {df['Date'].dt.year.min()} → {df['Date'].dt.year.max()}")


print(f"\nTrain: {len(train_df)} earthquakes (up to {SPLIT_YEAR-1})")
print(f"Test : {len(test_df)}  earthquakes ({SPLIT_YEAR} onward)")

# ── Features ──────────────────────────────────────────────────────
features = [
    "Latitude", "Longitude", "Depth",
    "lat_sin", "lat_cos", "lon_sin", "lon_cos",
    "plate_distance", "plate_score",
    "zone_count", "zone_avg_mag", "zone_max_mag",
    "zone_std_mag", "zone_avg_depth", "zone_rate", "zone_avg_plate"
]

X_train = train_df[features]
X_test  = test_df[features]

# ═══════════════════════════════════════════════════════════════════
# MODEL 1 — Magnitude Regressor
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  MODEL 1: Magnitude Regressor")
print("="*55)

y_train_mag = train_df["Magnitude"]
y_test_mag  = test_df["Magnitude"]

mag_model = xgb.XGBRegressor(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.04,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    random_state=42,
    verbosity=0
)
mag_model.fit(X_train, y_train_mag, verbose=False)

y_pred_mag = mag_model.predict(X_test)
mae = mean_absolute_error(y_test_mag, y_pred_mag)
r2  = r2_score(y_test_mag, y_pred_mag)

print(f"  MAE : {mae:.3f} magnitude units")
print(f"  R²  : {r2:.3f}")

# Cross validation
cv = cross_val_score(mag_model, X_train, y_train_mag, cv=5, scoring="r2")
print(f"  Cross-val R²: {cv.mean():.3f} ± {cv.std():.3f}")

print(f"\n  Plain English: our model predicts magnitude")
print(f"  within ±{mae:.2f} units on average.")
print(f"  e.g. real quake is 6.5 → we predict {6.5-mae:.2f}–{6.5+mae:.2f}")

print("\n  Sample predictions:")
print("  Predicted | Actual | Off by | Grade")
print("  ─────────────────────────────────────")
for pred, actual in list(zip(y_pred_mag[:12], y_test_mag.values[:12])):
    diff = abs(pred - actual)
    grade = "✓ Good" if diff < 0.3 else ("~ OK" if diff < 0.5 else "✗ Miss")
    print(f"  {pred:.2f}      | {actual:.2f}   | {diff:.2f}   | {grade}")

# ═══════════════════════════════════════════════════════════════════
# MODEL 2 — Hazard Level Classifier
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  MODEL 2: Seismic Hazard Classifier")
print("="*55)

le = LabelEncoder()
y_train_haz = le.fit_transform(train_df["hazard"])
y_test_haz  = le.transform(test_df["hazard"])

print(f"  Classes: {list(le.classes_)}")
print(f"  Train distribution: {train_df['hazard'].value_counts().to_dict()}")

# Handle class imbalance
counts = np.bincount(y_train_haz)
weights = len(y_train_haz) / (len(counts) * counts)
sample_weights = np.array([weights[y] for y in y_train_haz])

haz_model = xgb.XGBClassifier(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.04,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="mlogloss",
    random_state=42,
    verbosity=0
)
haz_model.fit(X_train, y_train_haz, sample_weight=sample_weights)

y_pred_haz = haz_model.predict(X_test)
print("\n  Classification Report:")
print(classification_report(
    y_test_haz, y_pred_haz,
    target_names=le.classes_,
    zero_division=0
))

# ═══════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE — what did the model actually learn?
# ═══════════════════════════════════════════════════════════════════
print("="*55)
print("  What the model learned (feature importance)")
print("="*55)
print("  These are the features that matter most:\n")

importances = mag_model.feature_importances_
sorted_feats = sorted(zip(features, importances),
                      key=lambda x: x[1], reverse=True)

for feat, imp in sorted_feats:
    bar = "█" * int(imp * 70)
    print(f"  {feat:20s} {bar} {imp:.3f}")

# ═══════════════════════════════════════════════════════════════════
# SAVE EVERYTHING
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  Saving models...")
print("="*55)

os.makedirs("models", exist_ok=True)
joblib.dump(mag_model, "models/magnitude_model.pkl")
joblib.dump(haz_model, "models/hazard_model.pkl")
joblib.dump(le,        "models/label_encoder.pkl")
joblib.dump(features,  "models/feature_names.pkl")

zone_stats = pd.read_csv("data/zone_stats.csv")
joblib.dump(zone_stats, "models/zone_stats.pkl")

print("  models/magnitude_model.pkl  ✓")
print("  models/hazard_model.pkl     ✓")
print("  models/label_encoder.pkl    ✓")
print("  models/feature_names.pkl    ✓")
print("  models/zone_stats.pkl       ✓")
print("\nAll done! Models are ready for the Streamlit app. ✓")

