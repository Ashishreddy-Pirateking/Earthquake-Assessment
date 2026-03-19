import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, classification_report
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib
import os

print("Loading cleaned data...")
df = pd.read_csv("data/cleaned_data.csv")
print(f"Loaded {len(df)} rows")

# ═══════════════════════════════════════════════════════════════════
# THE CORRECT APPROACH — TEMPORAL SPLIT
#
# Real world: you train on PAST earthquakes, predict FUTURE ones.
# So we split by DATE, not randomly.
# Train = 1965 to 2010
# Test  = 2011 to 2016 (model has never seen these)
# This is how real seismologists evaluate models.
# ═══════════════════════════════════════════════════════════════════

# Parse dates properly
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

print(f"\nDate range: {df['Date'].min().date()} → {df['Date'].max().date()}")

# Build zone stats using ONLY training data (no leakage!)
# Training data = before 2011
train_df = df[df["Date"].dt.year < 2011].copy()
test_df  = df[df["Date"].dt.year >= 2011].copy()

print(f"Training period (1965–2010): {len(train_df)} earthquakes")
print(f"Test period     (2011–2016): {len(test_df)} earthquakes")

# ── Build zone features from TRAINING DATA ONLY ───────────────────
print("\nBuilding zone features from training data only...")

train_df["lat_bin"] = (train_df["Latitude"]  / 2).round() * 2
train_df["lon_bin"] = (train_df["Longitude"] / 2).round() * 2

zone_stats = train_df.groupby(["lat_bin", "lon_bin"]).agg(
    zone_count   = ("Magnitude", "count"),
    zone_avg_mag = ("Magnitude", "mean"),
    zone_max_mag = ("Magnitude", "max"),
    zone_std_mag = ("Magnitude", "std"),
).reset_index()
zone_stats["zone_std_mag"] = zone_stats["zone_std_mag"].fillna(0)

# Apply zone stats to BOTH train and test
for d in [train_df, test_df]:
    d["lat_bin"] = (d["Latitude"]  / 2).round() * 2
    d["lon_bin"] = (d["Longitude"] / 2).round() * 2

train_df = train_df.merge(zone_stats, on=["lat_bin","lon_bin"], how="left")
test_df  = test_df.merge(zone_stats,  on=["lat_bin","lon_bin"], how="left")

# Fill zones that appear in test but not train with global averages
for col in ["zone_count","zone_avg_mag","zone_max_mag","zone_std_mag"]:
    global_avg = train_df[col].mean()
    train_df[col] = train_df[col].fillna(global_avg)
    test_df[col]  = test_df[col].fillna(global_avg)

# ── Features ──────────────────────────────────────────────────────
features = [
    "Latitude", "Longitude", "Depth",
    "lat_sin", "lat_cos", "lon_sin", "lon_cos",
    "zone_count", "zone_avg_mag", "zone_max_mag", "zone_std_mag"
]

X_train = train_df[features]
X_test  = test_df[features]

# ═══════════════════════════════════════════════════════════════════
# MODEL 1 — Magnitude Regressor
# ═══════════════════════════════════════════════════════════════════
print("\n── Model 1: Magnitude Regressor ──")

y_train_mag = train_df["Magnitude"]
y_test_mag  = test_df["Magnitude"]

mag_model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)
mag_model.fit(X_train, y_train_mag, verbose=False)

y_pred_mag = mag_model.predict(X_test)
mae = mean_absolute_error(y_test_mag, y_pred_mag)
r2  = r2_score(y_test_mag, y_pred_mag)

print(f"  MAE : {mae:.3f} magnitude units")
print(f"  R²  : {r2:.3f}")
print(f"\n  What MAE means in plain English:")
print(f"  On average, our prediction is off by {mae:.2f} magnitude units.")
print(f"  e.g. if real quake is 6.5, we predict between")
print(f"  {6.5-mae:.2f} and {6.5+mae:.2f} — that is actually useful!")

print("\n  Sample predictions vs actual:")
print("  Predicted | Actual | Off by")
print("  ─────────────────────────────")
for pred, actual in list(zip(y_pred_mag[:10], y_test_mag.values[:10])):
    diff = abs(pred - actual)
    ok = "✓" if diff < 0.4 else "~"
    print(f"  {pred:.2f}      | {actual:.2f}   | {diff:.2f}  {ok}")

# ═══════════════════════════════════════════════════════════════════
# MODEL 2 — Hazard Classifier (built honestly)
# High hazard = shallow + historically very active zone
# ═══════════════════════════════════════════════════════════════════
print("\n── Model 2: Hazard Level Classifier ──")

def hazard_level(row):
    if row["zone_count"] > 150 and row["Depth"] < 70:
        return "high"
    elif row["zone_count"] > 50:
        return "moderate"
    else:
        return "low"

train_df["hazard"] = train_df.apply(hazard_level, axis=1)
test_df["hazard"]  = test_df.apply(hazard_level, axis=1)

le = LabelEncoder()
le.fit(train_df["hazard"])

y_train_haz = le.transform(train_df["hazard"])
y_test_haz  = le.transform(test_df["hazard"])

print(f"  Hazard distribution in training data:")
print(f"  {train_df['hazard'].value_counts().to_dict()}")

counts = np.bincount(y_train_haz)
weights = len(y_train_haz) / (len(counts) * counts)
sample_weights = np.array([weights[y] for y in y_train_haz])

haz_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
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

# ── Feature Importance ────────────────────────────────────────────
print("── Feature importance ──")
importances = mag_model.feature_importances_
for feat, imp in sorted(zip(features, importances),
                        key=lambda x: x[1], reverse=True):
    bar = "█" * int(imp * 60)
    print(f"  {feat:20s} {bar} {imp:.3f}")

# ── Save everything ───────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
joblib.dump(mag_model,  "models/magnitude_model.pkl")
joblib.dump(haz_model,  "models/hazard_model.pkl")
joblib.dump(le,         "models/label_encoder.pkl")
joblib.dump(features,   "models/feature_names.pkl")
joblib.dump(zone_stats, "models/zone_stats.pkl")

# Save final data for the app
full_df = pd.concat([train_df, test_df]).reset_index(drop=True)
full_df.to_csv("data/final_data.csv", index=False)

print("\nAll models saved to models/ folder!")
print("Done! ✓")
