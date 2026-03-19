import pandas as pd
import numpy as np

# ── Step 1: Load the data ──────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("data/database.csv")
print(f"Loaded {len(df)} rows and {len(df.columns)} columns")

# ── Step 2: Keep only earthquake events (not explosions etc.) ──────
df = df[df["Type"] == "Earthquake"].copy()
print(f"After keeping only earthquakes: {len(df)} rows")

# ── Step 3: Keep only the columns we actually need ─────────────────
df = df[["Date", "Time", "Latitude", "Longitude", "Depth", "Magnitude"]].copy()

# ── Step 4: Fix missing values ─────────────────────────────────────
# Drop any rows where Latitude, Longitude, Depth or Magnitude is missing
df = df.dropna(subset=["Latitude", "Longitude", "Depth", "Magnitude"])
print(f"After removing missing values: {len(df)} rows")

# ── Step 5: Add new smart features ────────────────────────────────
# Depth category (shallow quakes cause more damage than deep ones)
def depth_category(d):
    if d < 70:
        return "shallow"
    elif d < 300:
        return "intermediate"
    else:
        return "deep"

df["depth_category"] = df["Depth"].apply(depth_category)

# Magnitude category (this is our target — what we want to predict)
def mag_category(m):
    if m < 6.0:
        return "moderate"    # 5.5 - 5.9
    elif m < 7.0:
        return "strong"      # 6.0 - 6.9
    elif m < 8.0:
        return "major"       # 7.0 - 7.9
    else:
        return "great"       # 8.0+

df["mag_category"] = df["Magnitude"].apply(mag_category)

# Convert lat/lon to cyclical features (so the model understands
# that -180 and +180 longitude are actually next to each other)
df["lat_sin"] = np.sin(np.radians(df["Latitude"]))
df["lat_cos"] = np.cos(np.radians(df["Latitude"]))
df["lon_sin"] = np.sin(np.radians(df["Longitude"]))
df["lon_cos"] = np.cos(np.radians(df["Longitude"]))

# ── Step 6: Save the cleaned data ─────────────────────────────────
df.to_csv("data/cleaned_data.csv", index=False)
print("\nCleaned data saved to data/cleaned_data.csv")
print(f"\nFinal shape: {df.shape}")
print(f"\nMagnitude category counts:")
print(df["mag_category"].value_counts())
print(f"\nDepth category counts:")
print(df["depth_category"].value_counts())
print("\nDone! ✓")




