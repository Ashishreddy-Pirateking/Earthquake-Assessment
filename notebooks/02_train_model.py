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
print(f"Loaded {len(df)} rows")

# ── Features we'll use to train ───────────────────────────────────
# These are the inputs the model learns from
features = [
    "Latitude",
    "Longitude",
    "Depth",
    "lat_sin",
    "lat_cos",
    "lon_sin",
    "lon_cos"
]

# ── Target 1: Magnitude (a number — regression problem) ───────────
# We want to predict the actual magnitude value e.g. 6.3, 7.1
X = df[features]
y_magnitude = df["Magnitude"]

# Split into training set (80%) and test set (20%)
# Training = model learns from this
# Test = we check how well it learned on data it never saw
X_train, X_test, y_train, y_test = train_test_split(
    X, y_magnitude, test_size=0.2, random_state=42
)

print(f"\nTraining set: {len(X_train)} rows")
print(f"Test set:     {len(X_test)} rows")

# ── Train Model 1: Magnitude Regressor ───────────────────────────
print("\nTraining magnitude prediction model...")
mag_model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)
mag_model.fit(X_train, y_train)

# Check how good it is
y_pred = mag_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
print(f"  Mean Absolute Error : {mae:.3f}  (lower is better)")
print(f"  R² Score            : {r2:.3f}   (closer to 1.0 is better)")

# ── Target 2: Severity Class (a label — classification problem) ───
# We want to predict moderate / strong / major / great
y_severity = df["mag_category"]

# Encode the labels to numbers (model needs numbers not words)
le = LabelEncoder()
y_severity_encoded = le.fit_transform(y_severity)

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X, y_severity_encoded, test_size=0.2, random_state=42
)

# ── Train Model 2: Severity Classifier ───────────────────────────
print("\nTraining severity classifier model...")
sev_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="mlogloss",
    random_state=42,
    verbosity=0
)
sev_model.fit(X_train2, y_train2)

# Check how good it is
y_pred2 = sev_model.predict(X_test2)
print("\nSeverity Classification Report:")
print(classification_report(
    y_test2, y_pred2,
    target_names=le.classes_
))

# ── Save both models ──────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
joblib.dump(mag_model, "models/magnitude_model.pkl")
joblib.dump(sev_model, "models/severity_model.pkl")
joblib.dump(le,        "models/label_encoder.pkl")
joblib.dump(features,  "models/feature_names.pkl")

print("\nAll models saved to models/ folder")
print("Done! ✓")
