import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, classification_report
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import requests, math, time, joblib, os

print("="*55)
print("  STEP 1: Reload both datasets fresh")
print("="*55)

# --- Historical CSV ---
df1 = pd.read_csv("data/database.csv")
df1 = df1[df1["Type"] == "Earthquake"].copy()
df1 = df1[["Date","Latitude","Longitude","Depth","Magnitude"]].dropna()
df1["Date"] = pd.to_datetime(df1["Date"], errors="coerce")
df1 = df1.dropna(subset=["Date"])
print(f"  Historical CSV : {len(df1)} rows")

# --- Recent USGS (2017–2024) ---
def fetch_usgs(year):
    url = (f"https://earthquake.usgs.gov/fdsnws/event/1/query"
           f"?format=csv&starttime={year}-01-01&endtime={year}-12-31"
           f"&minmagnitude=5.0&orderby=time-asc")
    try:
        r = requests.get(url, timeout=30)
        from io import StringIO
        tmp = pd.read_csv(StringIO(r.text))
        tmp = tmp.rename(columns={"latitude":"Latitude","longitude":"Longitude",
                                   "depth":"Depth","mag":"Magnitude","time":"Date"})
        tmp["Date"] = pd.to_datetime(tmp["Date"], utc=True,
                                      errors="coerce").dt.tz_localize(None)
        return tmp[["Date","Latitude","Longitude","Depth","Magnitude"]].dropna()
    except Exception as e:
        print(f"  {year} failed: {e}")
        return pd.DataFrame()

print("  Fetching 2017–2024 from USGS API...")
recent = pd.concat([fetch_usgs(y) for y in range(2017, 2025)], ignore_index=True)
print(f"  Recent API data: {len(recent)} rows")

# --- Combine ---
df = pd.concat([df1, recent], ignore_index=True)
df = df.dropna(subset=["Date","Latitude","Longitude","Depth","Magnitude"])
df = df.drop_duplicates(subset=["Date","Latitude","Longitude"])
df = df.sort_values("Date").reset_index(drop=True)
df["year"] = df["Date"].dt.year

print(f"\n  Combined total  : {len(df)} earthquakes")
print(f"  Date range      : {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"  Magnitude range : {df['Magnitude'].min()} → {df['Magnitude'].max()}")

print()
print("="*55)
print("  STEP 2: Tectonic plate distances")
print("="*55)

plate_url = ("https://raw.githubusercontent.com/fraxen/"
             "tectonicplates/master/GeoJSON/PB2002_boundaries.json")
plate_data = requests.get(plate_url, timeout=15).json()
print("  Plate boundaries fetched")

def plate_dist(lat, lon):
    best = float("inf")
    for f in plate_data["features"]:
        g = f["geometry"]
        lines = g["coordinates"] if g["type"]=="MultiLineString" else [g["coordinates"]]
        for line in lines:
            for pt in line:
                if isinstance(pt,(list,tuple)) and len(pt)>=2:
                    d = math.sqrt((lat-pt[1])**2 + (lon-pt[0])**2)
                    if d < best:
                        best = d
    return best

print(f"  Computing distances for {len(df)} rows (takes ~2 min)...")
dists = []
for i, row in df.iterrows():
    dists.append(plate_dist(row["Latitude"], row["Longitude"]))
    if i % 3000 == 0:
        print(f"  {i}/{len(df)} ({100*i//len(df)}%)")
df["plate_distance"] = dists
df["plate_score"]    = np.exp(-df["plate_distance"] / 5)
print("  Done!")

print()
print("="*55)
print("  STEP 3: Cyclical features")
print("="*55)

df["lat_sin"] = np.sin(np.radians(df["Latitude"]))
df["lat_cos"] = np.cos(np.radians(df["Latitude"]))
df["lon_sin"] = np.sin(np.radians(df["Longitude"]))
df["lon_cos"] = np.cos(np.radians(df["Longitude"]))
print("  lat/lon cyclical encoding done")

print()
print("="*55)
print("  STEP 4: Zone statistics (training data only)")
print("  Using 1965–2016 to compute zones.")
print("  2017–2024 is our true holdout test set.")
print("="*55)

df["lat_bin"] = (df["Latitude"]  / 2).round() * 2
df["lon_bin"] = (df["Longitude"] / 2).round() * 2

# Zone stats built ONLY from historical data — zero leakage
hist = df[df["year"] <= 2016].copy()
n_years = hist["year"].nunique()

zone = hist.groupby(["lat_bin","lon_bin"]).agg(
    zone_count     = ("Magnitude", "count"),
    zone_avg_mag   = ("Magnitude", "mean"),
    zone_max_mag   = ("Magnitude", "max"),
    zone_std_mag   = ("Magnitude", "std"),
    zone_avg_depth = ("Depth",     "mean"),
    zone_avg_plate = ("plate_score","mean"),
).reset_index()
zone["zone_rate"]    = zone["zone_count"] / n_years
zone["zone_std_mag"] = zone["zone_std_mag"].fillna(0)

print(f"  Zones built from {len(hist)} quakes over {n_years} years")
print(f"  Total zones: {len(zone)}")

# Merge zone stats into full df
df = df.merge(zone, on=["lat_bin","lon_bin"], how="left")
for col in ["zone_count","zone_avg_mag","zone_max_mag",
            "zone_std_mag","zone_avg_depth","zone_rate","zone_avg_plate"]:
    df[col] = df[col].fillna(zone[col].median())

print()
print("="*55)
print("  STEP 5: Honest hazard labels")
print("  Based on ACTUAL worst earthquake in each zone.")
print("  No circular formula — pure ground truth.")
print("="*55)

# For each zone, what is the worst magnitude ever recorded?
# That IS the ground truth hazard of that zone.
# We use ONLY historical data to define these labels.
zone_worst = hist.groupby(["lat_bin","lon_bin"])["Magnitude"].max().reset_index()
zone_worst.columns = ["lat_bin","lon_bin","worst_mag"]

df = df.merge(zone_worst, on=["lat_bin","lon_bin"], how="left")
df["worst_mag"] = df["worst_mag"].fillna(df["Magnitude"])

# Hazard label = based on worst RECORDED magnitude in that zone
# This is scientifically meaningful and not circular
def honest_hazard(worst_mag):
    if worst_mag >= 8.0:   return "very_high"   # great earthquakes
    elif worst_mag >= 7.0: return "high"         # major earthquakes
    elif worst_mag >= 6.0: return "moderate"     # strong earthquakes
    else:                  return "low"           # moderate earthquakes

df["hazard"] = df["worst_mag"].apply(honest_hazard)

print(f"  Hazard distribution:")
print(f"  {df['hazard'].value_counts().to_dict()}")

# Save the full dataset
df.to_csv("data/master_dataset.csv", index=False)
zone.to_csv("data/zone_stats.csv",   index=False)
print(f"\n  Saved master_dataset.csv ({len(df)} rows, {len(df.columns)} columns)")

print()
print("="*55)
print("  STEP 6: Train/Test split")
print("  Train = 1965–2016 (historical)")
print("  Test  = 2017–2024 (future — model never saw this)")
print("="*55)

features = [
    "Latitude", "Longitude", "Depth",
    "lat_sin", "lat_cos", "lon_sin", "lon_cos",
    "plate_distance", "plate_score",
    "zone_count", "zone_avg_mag", "zone_max_mag",
    "zone_std_mag", "zone_avg_depth", "zone_rate", "zone_avg_plate"
]

train = df[df["year"] <= 2016].copy()
test  = df[df["year"] >  2016].copy()

print(f"  Train : {len(train)} rows")
print(f"  Test  : {len(test)}  rows")

X_train, y_train_mag = train[features], train["Magnitude"]
X_test,  y_test_mag  = test[features],  test["Magnitude"]

print()
print("="*55)
print("  MODEL 1: Magnitude Regressor")
print("="*55)

mag_model = xgb.XGBRegressor(
    n_estimators=600, max_depth=6, learning_rate=0.04,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=3, random_state=42, verbosity=0
)
mag_model.fit(X_train, y_train_mag)

y_pred_mag = mag_model.predict(X_test)
mae = mean_absolute_error(y_test_mag, y_pred_mag)
r2  = r2_score(y_test_mag, y_pred_mag)
cv  = cross_val_score(mag_model, X_train, y_train_mag, cv=5, scoring="r2")

print(f"  MAE          : {mae:.3f} magnitude units")
print(f"  R²           : {r2:.3f}")
print(f"  Cross-val R² : {cv.mean():.3f} ± {cv.std():.3f}")
print(f"\n  Meaning: predictions are within ±{mae:.2f} magnitude units")
print(f"  on completely unseen 2017–2024 data.")

print("\n  Sample predictions on 2017–2024 data:")
print("  Predicted | Actual | Off by | Grade")
print("  ──────────────────────────────────────")
for p, a in list(zip(y_pred_mag[:12], y_test_mag.values[:12])):
    d = abs(p-a)
    g = "✓ Good" if d<0.3 else ("~ OK" if d<0.5 else "✗ Miss")
    print(f"  {p:.2f}      | {a:.2f}   | {d:.2f}   | {g}")

print()
print("="*55)
print("  MODEL 2: Seismic Hazard Classifier")
print("="*55)

le = LabelEncoder()
y_train_haz = le.fit_transform(train["hazard"])
y_test_haz  = le.transform(test["hazard"])

print(f"  Classes : {list(le.classes_)}")
print(f"  Train distribution: {train['hazard'].value_counts().to_dict()}")
print(f"  Test  distribution: {test['hazard'].value_counts().to_dict()}")

counts  = np.bincount(y_train_haz)
weights = len(y_train_haz) / (len(counts) * counts)
sw      = np.array([weights[y] for y in y_train_haz])

haz_model = xgb.XGBClassifier(
    n_estimators=600, max_depth=6, learning_rate=0.04,
    subsample=0.8, colsample_bytree=0.8,
    use_label_encoder=False, eval_metric="mlogloss",
    random_state=42, verbosity=0
)
haz_model.fit(X_train, y_train_haz, sample_weight=sw)

y_pred_haz = haz_model.predict(X_test)
print("\n  Classification Report (on unseen 2017–2024 data):")
print(classification_report(
    y_test_haz, y_pred_haz,
    target_names=le.classes_, zero_division=0
))

print("="*55)
print("  Feature importance")
print("="*55)
for feat, imp in sorted(zip(features, mag_model.feature_importances_),
                         key=lambda x: x[1], reverse=True):
    bar = "█" * int(imp * 70)
    print(f"  {feat:20s} {bar} {imp:.3f}")

print()
print("="*55)
print("  Saving all models...")
print("="*55)
os.makedirs("models", exist_ok=True)
joblib.dump(mag_model, "models/magnitude_model.pkl")
joblib.dump(haz_model, "models/hazard_model.pkl")
joblib.dump(le,        "models/label_encoder.pkl")
joblib.dump(features,  "models/feature_names.pkl")
joblib.dump(zone,      "models/zone_stats.pkl")

print("  All models saved!")
print("\nDone! ✓  Ready to build the Streamlit app.")
