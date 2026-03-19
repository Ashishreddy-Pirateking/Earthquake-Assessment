import pandas as pd
import numpy as np
import requests
import math
import time
import os

print("=" * 55)
print("  STEP 1: Loading your existing USGS database.csv")
print("=" * 55)

df1 = pd.read_csv("data/database.csv")
df1 = df1[df1["Type"] == "Earthquake"].copy()
df1 = df1[["Date","Time","Latitude","Longitude","Depth","Magnitude"]].dropna()
df1["source"] = "usgs_historical"
print(f"  Loaded {len(df1)} rows from database.csv")

print()
print("=" * 55)
print("  STEP 2: Fetching recent USGS data (2017–2024)")
print("  This covers years missing from your CSV")
print("=" * 55)

def fetch_usgs_year(year):
    url = (
        f"https://earthquake.usgs.gov/fdsnws/event/1/query"
        f"?format=csv"
        f"&starttime={year}-01-01"
        f"&endtime={year}-12-31"
        f"&minmagnitude=5.0"
        f"&orderby=time-asc"
    )
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200 and len(r.content) > 100:
            from io import StringIO
            tmp = pd.read_csv(StringIO(r.text))
            tmp = tmp.rename(columns={
                "latitude":  "Latitude",
                "longitude": "Longitude",
                "depth":     "Depth",
                "mag":       "Magnitude",
                "time":      "Date"
            })
            tmp["Date"] = pd.to_datetime(tmp["Date"], errors="coerce")
            tmp = tmp[["Date","Latitude","Longitude","Depth","Magnitude"]].dropna()
            tmp["source"] = "usgs_recent"
            print(f"  {year}: fetched {len(tmp)} earthquakes")
            return tmp
        else:
            print(f"  {year}: no data returned")
            return None
    except Exception as e:
        print(f"  {year}: failed ({e})")
        return None

recent_frames = []
for year in range(2017, 2025):
    frame = fetch_usgs_year(year)
    if frame is not None:
        recent_frames.append(frame)
    time.sleep(1)  # be polite to the API

if recent_frames:
    df2 = pd.concat(recent_frames, ignore_index=True)
    print(f"\n  Total recent earthquakes fetched: {len(df2)}")
else:
    df2 = pd.DataFrame()
    print("  Could not fetch recent data — continuing with historical only")

print()
print("=" * 55)
print("  STEP 3: Fetching tectonic plate boundaries")
print("=" * 55)

plate_url = ("https://raw.githubusercontent.com/fraxen/"
             "tectonicplates/master/GeoJSON/PB2002_boundaries.json")
try:
    plate_data = requests.get(plate_url, timeout=15).json()
    print("  Tectonic plate data fetched successfully")
    HAS_PLATES = True
except:
    print("  Could not fetch plate data — will skip plate distance feature")
    HAS_PLATES = False

print()
print("=" * 55)
print("  STEP 4: Combining all data")
print("=" * 55)

df1["Date"] = pd.to_datetime(df1["Date"], errors="coerce")

frames_to_combine = [df1]
if len(df2) > 0:
    frames_to_combine.append(df2)

df = pd.concat(frames_to_combine, ignore_index=True)
df = df.dropna(subset=["Latitude","Longitude","Depth","Magnitude","Date"])
df = df.drop_duplicates(subset=["Date","Latitude","Longitude"])
df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce").dt.tz_localize(None)
df = df.sort_values("Date").reset_index(drop=True)

print(f"  Combined dataset: {len(df)} earthquakes")
print(f"  Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"  Magnitude range: {df['Magnitude'].min()} → {df['Magnitude'].max()}")

print()
print("=" * 55)
print("  STEP 5: Computing plate distance for every earthquake")
print("  (This will take 1-2 minutes — please wait)")
print("=" * 55)

def distance_to_plate(lat, lon, plate_data):
    min_dist = float("inf")
    for feature in plate_data["features"]:
        geom = feature["geometry"]
        lines = (geom["coordinates"]
                 if geom["type"] == "MultiLineString"
                 else [geom["coordinates"]])
        for line in lines:
            for point in line:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    d = math.sqrt((lat - point[1])**2 + (lon - point[0])**2)
                    if d < min_dist:
                        min_dist = d
    return min_dist

if HAS_PLATES:
    distances = []
    total = len(df)
    for i, row in df.iterrows():
        d = distance_to_plate(row["Latitude"], row["Longitude"], plate_data)
        distances.append(d)
        if i % 2000 == 0:
            print(f"  Progress: {i}/{total} ({100*i//total}%)")
    df["plate_distance"] = distances
    print(f"  Done computing plate distances!")
else:
    df["plate_distance"] = 5.0  # fallback average

print()
print("=" * 55)
print("  STEP 6: Feature Engineering")
print("=" * 55)

# Cyclical encoding of lat/lon
df["lat_sin"] = np.sin(np.radians(df["Latitude"]))
df["lat_cos"] = np.cos(np.radians(df["Latitude"]))
df["lon_sin"] = np.sin(np.radians(df["Longitude"]))
df["lon_cos"] = np.cos(np.radians(df["Longitude"]))

# Plate proximity score (0 to 1, higher = closer to plate boundary)
df["plate_score"] = np.exp(-df["plate_distance"] / 5)

# Depth category
df["depth_bin"] = pd.cut(
    df["Depth"],
    bins=[0, 70, 300, 700],
    labels=["shallow", "intermediate", "deep"]
)

# Year and decade
df["year"]   = df["Date"].dt.year
df["decade"] = (df["year"] // 10) * 10

print(f"  Features added: lat_sin, lat_cos, lon_sin, lon_cos,")
print(f"                  plate_distance, plate_score, depth_bin")

print()
print("=" * 55)
print("  STEP 7: Building zone-level seismicity features")
print("  (The key to making our model actually work)")
print("=" * 55)

# Use only pre-2015 data to compute zone stats
# This prevents ANY leakage into our test period
historical = df[df["year"] < 2015].copy()

historical["lat_bin"] = (historical["Latitude"] / 2).round() * 2
historical["lon_bin"] = (historical["Longitude"] / 2).round() * 2

n_years = historical["year"].nunique()
print(f"  Using {len(historical)} earthquakes over {n_years} years to build zones")

zone_stats = historical.groupby(["lat_bin","lon_bin"]).agg(
    zone_count    = ("Magnitude", "count"),
    zone_avg_mag  = ("Magnitude", "mean"),
    zone_max_mag  = ("Magnitude", "max"),
    zone_std_mag  = ("Magnitude", "std"),
    zone_avg_depth= ("Depth",     "mean"),
    zone_avg_plate= ("plate_score","mean"),
).reset_index()

zone_stats["zone_rate"]    = zone_stats["zone_count"] / n_years
zone_stats["zone_std_mag"] = zone_stats["zone_std_mag"].fillna(0)

print(f"  Created stats for {len(zone_stats)} geographic zones")
print(f"  Zone seismicity rate (quakes/year): "
      f"min={zone_stats['zone_rate'].min():.2f}, "
      f"max={zone_stats['zone_rate'].max():.2f}, "
      f"mean={zone_stats['zone_rate'].mean():.2f}")

# Merge zone stats into full dataset
df["lat_bin"] = (df["Latitude"] / 2).round() * 2
df["lon_bin"] = (df["Longitude"] / 2).round() * 2
df = df.merge(zone_stats, on=["lat_bin","lon_bin"], how="left")

# Fill any zones not in training history with global averages
for col in ["zone_count","zone_avg_mag","zone_max_mag",
            "zone_std_mag","zone_avg_depth","zone_rate","zone_avg_plate"]:
    df[col] = df[col].fillna(df[col].median())

# Build our prediction target: seismic hazard label
def hazard_label(row):
    rate  = row["zone_rate"]
    plate = row["plate_score"]
    depth = row["Depth"]
    if rate >= 3.0 and depth < 70 and plate > 0.5:
        return "very_high"
    elif rate >= 1.5 or (plate > 0.7 and depth < 70):
        return "high"
    elif rate >= 0.3 or plate > 0.4:
        return "moderate"
    else:
        return "low"

df["hazard"] = df.apply(hazard_label, axis=1)

print(f"\n  Hazard label distribution:")
print(f"  {df['hazard'].value_counts().to_dict()}")

print()
print("=" * 55)
print("  STEP 8: Saving everything")
print("=" * 55)

os.makedirs("data", exist_ok=True)
df.to_csv("data/master_dataset.csv", index=False)
zone_stats.to_csv("data/zone_stats.csv", index=False)

print(f"  Saved data/master_dataset.csv  ({len(df)} rows)")
print(f"  Saved data/zone_stats.csv      ({len(zone_stats)} zones)")
print()
print("  Summary:")
print(f"  Total earthquakes : {len(df)}")
print(f"  Date range        : {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"  Features ready    : {len(df.columns)} columns")
print()
print("Done! ✓  Ready for model training.")
