import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, classification_report
import xgboost as xgb
import joblib
import os

print("Loading cleaned data...")
df = pd.read_csv("data/cleaned_data.csv")

# ── Features ──────────────────────────────────────────────────────
features = [
    "Latitude", "Longitude", "Depth",
    "lat_sin", "lat_cos", "lon_sin", "lon_cos"
]

X = df[features]

# ═══════════════════════════════════════════════════════════════════
# MODEL 1 — Magnitude Regressor (improved)
# ═══════════════════════════════════════════════════════════════════
print("\n── Model 1: Magnitude Regressor ──")

y_mag = df["Magnitude"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_mag, test_size=0.2, random_state=42
)

# Add more trees, tune depth — helps capture regional patterns
mag_model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    random_state=42,
    verbosity=0
)
mag_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

y_pred = mag_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
print(f"  Mean Absolute Error : {mae:.3f}")
print(f"  R² Score            : {r2:.3f}")

# Show some example predictions vs reality
print("\n  Sample predictions vs actual:")
print("  Predicted | Actual")
print("  ──────────────────")
for pred, actual in list(zip(y_pred[:8], y_test.values[:8])):
    print(f"  {pred:.2f}      | {actual:.2f}")

# ═══════════════════════════════════════════════════════════════════
# MODEL 2 — Severity Classifier (fixed class imbalance)
# ═══════════════════════════════════════════════════════════════════
print("\n── Model 2: Severity Classifier ──")

# The fix: we tell the model to pay MORE attention to rare classes
# by calculating a weight for each class
y_sev = df["mag_category"]
le = LabelEncoder()
y_encoded = le.fit_transform(y_sev)

print(f"  Classes: {list(le.classes_)}")

# Count how many of each class we have
counts = df["mag_category"].value_counts()
print(f"\n  Class distribution:")
for cat, count in counts.items():
    print(f"    {cat:12s}: {count:6d} samples")

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
# stratify=y_encoded means: make sure test set has same
# proportion of each class as the full dataset

# Calculate class weights — rarer classes get higher weight
# so the model can't just ignore them
total = len(y_train2)
class_counts = np.bincount(y_train2)
class_weights = total / (len(class_counts) * class_counts)
sample_weights = np.array([class_weights[y] for y in y_train2])

print(f"\n  Class weights (higher = model pays more attention):")
for cls, w in zip(le.classes_, class_weights):
    print(f"    {cls:12s}: {w:.2f}")

sev_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="mlogloss",
    random_state=42,
    verbosity=0
)
sev_model.fit(
    X_train2, y_train2,
    sample_weight=sample_weights,
    verbose=False
)

y_pred2 = sev_model.predict(X_test2)
print("\n  Classification Report:")
print(classification_report(
    y_test2, y_pred2,
    target_names=le.classes_,
    zero_division=0
))

# ── Feature Importance (what does the model rely on most?) ────────
print("── Feature Importance ──")
importances = mag_model.feature_importances_
for feat, imp in sorted(zip(features, importances),
                         key=lambda x: x[1], reverse=True):
    bar = "█" * int(imp * 50)
    print(f"  {feat:15s} {bar} {imp:.3f}")

# ── Save everything ───────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
joblib.dump(mag_model, "models/magnitude_model.pkl")
joblib.dump(sev_model, "models/severity_model.pkl")
joblib.dump(le,        "models/label_encoder.pkl")
joblib.dump(features,  "models/feature_names.pkl")

print("\nAll improved models saved!")
print("Done! ✓")

