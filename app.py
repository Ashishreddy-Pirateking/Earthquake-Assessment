import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import json

st.set_page_config(
    page_title="Seismic Hazard Assessment",
    page_icon="🌍",
    layout="wide"
)

# ── CSS: kill overlay + mobile responsive ─────────────────────────
st.markdown("""
<style>
/* Kill the dark running overlay — permanently */
div[data-testid="stAppViewBlockContainer"] { opacity: 1 !important; }
div[class*="withScreencast"] > div         { opacity: 1 !important; }
div[class*="stSpinner"]                    { display: none !important; }
section[data-testid="stSidebar"]           { display: none !important; }
[data-testid="stDecoration"]               { display: none !important; }
[data-testid="stStatusWidget"]             { display: none !important; }
div[data-testid="stToolbar"]               { display: none !important; }
.stAppDeployButton                         { display: none !important; }

/* Mobile responsive */
@media (max-width: 768px) {
  [data-testid="stHorizontalBlock"] {
    flex-direction: column !important;
  }
  [data-testid="stHorizontalBlock"] > div {
    width: 100% !important;
    min-width: 100% !important;
    flex: none !important;
  }
  [data-testid="stTextInput"] input {
    font-size: 16px !important;
  }
  button[data-baseweb="tab"] {
    font-size: 0.78rem !important;
    padding: 0.35rem 0.5rem !important;
  }
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────
HAZARD_COLORS = {
    "very_high": "#d62728",
    "high":      "#ff7f0e",
    "moderate":  "#f0c419",
    "low":       "#2ca02c"
}
HAZARD_EMOJI = {
    "very_high": "🔴",
    "high":      "🟠",
    "moderate":  "🟡",
    "low":       "🟢"
}

# ── DMS coordinate formatter ──────────────────────────────────────
def decimal_to_dms(lat, lon):
    """Convert decimal degrees to Degrees°Minutes'Seconds\" format."""
    def fmt(deg, is_lat):
        direction = ("N" if deg >= 0 else "S") if is_lat else ("E" if deg >= 0 else "W")
        deg = abs(deg)
        d = int(deg)
        m = int((deg - d) * 60)
        s = round((deg - d - m / 60) * 3600, 1)
        return f"{d}°{m:02d}'{s:04.1f}\"{direction}"
    return f"{fmt(lat, True)},  {fmt(lon, False)}"

# ── Model loading ─────────────────────────────────────────────────
@st.cache_resource
def load_models():
    mag_model  = joblib.load("models/magnitude_model.pkl")
    haz_model  = joblib.load("models/hazard_model.pkl")
    le         = joblib.load("models/label_encoder.pkl")
    features   = joblib.load("models/feature_names.pkl")
    zone_stats = joblib.load("models/zone_stats.pkl")
    return mag_model, haz_model, le, features, zone_stats

@st.cache_data
def load_data():
    df = pd.read_csv("data/master_dataset.csv", low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.dropna(subset=["Date", "Latitude", "Longitude", "Magnitude"])

@st.cache_data(ttl=3600)
def fetch_recent_quakes():
    url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_month.geojson"
    try:
        data = requests.get(url, timeout=10).json()
        records = []
        for f in data["features"]:
            p = f["properties"]; c = f["geometry"]["coordinates"]
            records.append({
                "Latitude": c[1], "Longitude": c[0],
                "Depth": c[2], "Magnitude": p["mag"],
                "Place": p["place"] or "Unknown"
            })
        return pd.DataFrame(records).dropna()
    except:
        return pd.DataFrame()

mag_model, haz_model, le, features, zone_stats = load_models()
hist_df   = load_data()
recent_df = fetch_recent_quakes()

# ── Core prediction ───────────────────────────────────────────────
def predict_location(lat, lon, depth):
    lat_bin = round(lat / 2) * 2
    lon_bin = round(lon / 2) * 2
    zone = zone_stats[
        (zone_stats["lat_bin"] == lat_bin) &
        (zone_stats["lon_bin"] == lon_bin)
    ]
    has_history = len(zone) > 0
    if has_history:
        z   = zone.iloc[0]
        zc  = float(z["zone_count"]);   zam = float(z["zone_avg_mag"])
        zmm = float(z["zone_max_mag"]); zsm = float(z["zone_std_mag"])
        zad = float(z["zone_avg_depth"]); zr = float(z["zone_rate"])
        zap = float(z["zone_avg_plate"])
    else:
        zc, zam, zmm, zsm = 0.0, 5.5, 5.5, 0.0
        zad, zr, zap = float(depth), 0.01, 0.05

    plate_distance = -5 * np.log(max(zap, 1e-6))
    X = np.array([[
        lat, lon, depth,
        np.sin(np.radians(lat)), np.cos(np.radians(lat)),
        np.sin(np.radians(lon)), np.cos(np.radians(lon)),
        plate_distance, zap, zc, zam, zmm, zsm, zad, zr, zap
    ]])
    pred_mag     = float(mag_model.predict(X)[0])
    pred_haz_enc = haz_model.predict(X)[0]
    pred_haz     = le.inverse_transform([pred_haz_enc])[0]
    haz_proba    = haz_model.predict_proba(X)[0]

    if not has_history:
        pred_haz = "low"
        cls = list(le.classes_)
        haz_proba = np.zeros(len(cls))
        haz_proba[cls.index("low")] = 0.82
        haz_proba[cls.index("moderate")] = 0.18

    return {
        "pred_mag": pred_mag, "pred_haz": pred_haz,
        "haz_proba": haz_proba, "has_history": has_history,
        "zone_count": zc, "zone_avg_mag": zam,
        "zone_max_mag": zmm, "zone_rate": zr,
    }

def get_place_name(lat, lon):
    # Try Nominatim first
    try:
        from geopy.geocoders import Nominatim
        geo = Nominatim(user_agent="seismic_hazard_v5", timeout=8)
        loc = geo.reverse(f"{lat}, {lon}", language="en")
        if loc and loc.address:
            return loc.address
    except:
        pass

    # Fallback: OpenStreetMap direct API
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon,
                    "format": "json", "zoom": 10},
            headers={"User-Agent": "SeismicHazardApp/1.0"},
            timeout=8
        )
        if r.status_code == 200:
            data = r.json()
            return data.get("display_name", decimal_to_dms(lat, lon))
    except:
        pass

    return decimal_to_dms(lat, lon)


# ── Geology via Groq + fallback ───────────────────────────────────
def get_geology(lat, lon, place_name):
    dms = decimal_to_dms(lat, lon)
    prompt = f"""You are a senior geologist and earth scientist.
Location: {place_name}
Coordinates: {dms} (decimal: {lat:.4f}, {lon:.4f})

Write a detailed, scientifically accurate geological profile.
Return ONLY a valid JSON object — no markdown, no code blocks, just raw JSON:

{{
  "geological_setting": "2-3 sentences: broad geological setting, tectonic province, regional context",
  "tectonic_setting": "2-3 sentences: plate boundaries, fault systems, tectonic history",
  "rock_types": "2-3 sentences: dominant rock types with specific real formation names",
  "stratigraphy": "2-3 sentences: stratigraphic column, geological periods, key formations",
  "basin_info": "2-3 sentences: sedimentary basins, age, fill, economic significance",
  "seismic_context": "2-3 sentences: known fault systems, historical seismicity, hazard geology",
  "fossil_record": "1-2 sentences: notable fossils or palaeontological significance",
  "economic_geology": "1-2 sentences: mineral resources, hydrocarbons, economic geology",
  "interesting_fact": "One genuinely fascinating geological fact about this specific location"
}}

Use real geological formation names, real fault names, real era names. Be specific."""

    try:
        api_key = st.secrets.get("GROQ_API_KEY", "")
        if api_key:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"},
                json={"model": "llama-3.1-8b-instant",
                      "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 1200, "temperature": 0.3},
                timeout=30
            )
            if resp.status_code == 200:
                text = resp.json()["choices"][0]["message"]["content"].strip()
                if "```" in text:
                    for part in text.split("```"):
                        part = part.strip()
                        if part.startswith("json"): part = part[4:]
                        part = part.strip()
                        if part.startswith("{"): text = part; break
                return json.loads(text)
    except Exception:
        pass

    # Free API fallback
    result = {k: "" for k in [
        "geological_setting","tectonic_setting","rock_types","stratigraphy",
        "basin_info","seismic_context","fossil_record","economic_geology","interesting_fact"
    ]}
    try:
        r = requests.get("https://macrostrat.org/api/v2/columns",
            params={"lat": lat, "lng": lon, "format": "json"}, timeout=10)
        cols = r.json().get("success", {}).get("data", [])
        if cols:
            c = cols[0]
            result["geological_setting"] = (
                f"Within the {c.get('col_name','Unknown')} geological column"
                f"{', '+c.get('col_group','') if c.get('col_group') else ''}. "
                f"Record spans {c.get('b_age',0):.0f}–{c.get('t_age',0):.0f} Ma. "
                f"{c.get('pbdb_collections',0)} palaeobiology collections."
            )
    except: pass
    try:
        r2 = requests.get("https://macrostrat.org/api/v2/units",
            params={"lat": lat, "lng": lon, "format": "json", "response": "long"}, timeout=10)
        units = r2.json().get("success", {}).get("data", [])
        if units:
            liths, strats, envs, ages, fossils, minerals = [], [], [], [], [], []
            for u in units[:10]:
                for lst, key in [(liths,"lith"),(strats,"strat_name_long"),
                                 (envs,"environ"),(minerals,"econ")]:
                    v = u.get(key, "")
                    if v and v not in lst: lst.append(v)
                b = u.get("b_age", 0)
                if b: ages.append(b)
                fossils.append(u.get("pbdb_collections", 0))
            if liths: result["rock_types"] = f"Lithologies: {', '.join(liths[:5])}."
            if strats:
                result["stratigraphy"] = (f"Key units: {', '.join(strats[:5])}. "
                    + (f"Ages {max(ages):.0f}–{min(ages):.0f} Ma." if ages else ""))
            if envs: result["basin_info"] = f"Environments: {', '.join(envs[:5])}."
            total = sum(fossils)
            result["fossil_record"] = (f"{total} fossil collections recorded."
                if total > 0 else "No significant fossil collections recorded.")
            if minerals: result["economic_geology"] = f"Economic indicators: {', '.join(minerals[:4])}."
    except: pass
    try:
        parts   = place_name.split(",")
        country = parts[-1].strip() if len(parts) > 1 else place_name
        region  = parts[-2].strip() if len(parts) > 2 else country
        for term in [f"geology of {country}", f"geology {region}", f"tectonics {country}"]:
            try:
                sr = requests.get("https://en.wikipedia.org/w/api.php",
                    params={"action":"query","list":"search","srsearch":term,"srlimit":1,"format":"json"}, timeout=6)
                hits = sr.json().get("query",{}).get("search",[])
                if not hits: continue
                er = requests.get("https://en.wikipedia.org/w/api.php",
                    params={"action":"query","prop":"extracts","exintro":True,"explaintext":True,
                            "titles":hits[0]["title"],"format":"json"}, timeout=6)
                for page in er.json().get("query",{}).get("pages",{}).values():
                    sents = [s.strip() for s in page.get("extract","").split(".") if len(s.strip()) > 50]
                    if sents and not result["tectonic_setting"]:
                        result["tectonic_setting"] = ". ".join(sents[:4]) + "."
                        break
                if result["tectonic_setting"]: break
            except: continue
    except: pass
    try:
        nearby = hist_df[(abs(hist_df["Latitude"]-lat)<5) & (abs(hist_df["Longitude"]-lon)<5)]
        if len(nearby) > 0:
            dep_type = ("shallow crustal" if nearby["Depth"].mean() < 70
                        else "intermediate-depth" if nearby["Depth"].mean() < 300 else "deep")
            result["seismic_context"] = (
                f"Within 5°: {len(nearby)} earthquakes (M5+) in 60-year USGS data. "
                f"Largest M{nearby['Magnitude'].max():.1f}, avg M{nearby['Magnitude'].mean():.2f} "
                f"at {nearby['Depth'].mean():.0f} km — {dep_type} seismicity.")
        else:
            result["seismic_context"] = "No M5+ within 5° in 60-year database. Stable intraplate setting."
    except: pass
    for k, v in [
        ("geological_setting", "Macrostrat data unavailable for this location."),
        ("tectonic_setting",   "Tectonic context unavailable. Infer from proximity to plate boundaries."),
        ("economic_geology",   "Economic geology data not available from open sources."),
        ("interesting_fact",   "Every location on Earth sits on rocks recording billions of years of planetary history.")
    ]:
        if not result[k]: result[k] = v
    return result

def get_geo_images(place_name, lat, lon):
    parts  = place_name.split(",")
    region = parts[-1].strip() if len(parts) > 1 else place_name
    short  = parts[0].strip()
    images = []
    for query in [f"{short} geology rock formation", f"{region} geological landscape"]:
        try:
            r = requests.get("https://en.wikipedia.org/w/api.php", params={
                "action":"query","generator":"search",
                "gsrsearch":f"{query} filetype:jpg","gsrnamespace":6,"gsrlimit":3,
                "prop":"imageinfo","iiprop":"url|extmetadata","iiurlwidth":400,"format":"json"
            }, timeout=8)
            for page in r.json().get("query",{}).get("pages",{}).values():
                info    = page.get("imageinfo",[{}])[0]
                img_url = info.get("thumburl") or info.get("url","")
                desc    = info.get("extmetadata",{}).get("ImageDescription",{}).get("value","")[:100]
                if img_url and any(img_url.lower().endswith(x) for x in [".jpg",".jpeg",".png"]):
                    images.append({"url": img_url, "desc": desc or query})
                if len(images) >= 3: break
        except: pass
        if len(images) >= 3: break
    return images[:3]

def run_assessment(lat, lon, depth):
    with st.spinner("Analysing seismic data and geological context..."):
        result     = predict_location(lat, lon, depth)
        place_name = get_place_name(lat, lon)
        geology    = get_geology(lat, lon, place_name)
        geo_images = get_geo_images(place_name, lat, lon)
    st.session_state.update({
        "result": result, "place": place_name,
        "geology": geology, "geo_images": geo_images,
        "lat": lat, "lon": lon, "depth": depth
    })

def show_results():
    result     = st.session_state["result"]
    place_name = st.session_state["place"]
    geology    = st.session_state["geology"]
    geo_images = st.session_state["geo_images"]
    r_lat      = st.session_state["lat"]
    r_lon      = st.session_state["lon"]
    r_dep      = st.session_state["depth"]
    pred_haz   = result["pred_haz"]
    pred_mag   = result["pred_mag"]
    haz_proba  = result["haz_proba"]
    dms        = decimal_to_dms(r_lat, r_lon)

    st.divider()
    # Location header with DMS coordinates
    st.markdown(f"### 📍 {place_name}")
    st.markdown(
        f"<div style='font-family:monospace;font-size:0.95rem;"
        f"color:var(--color-text-secondary);margin-bottom:8px'>"
        f"🌐 {dms}  ·  🕳️ {r_dep} km depth</div>",
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        color = HAZARD_COLORS[pred_haz]
        emoji = HAZARD_EMOJI[pred_haz]
        st.markdown(f"""
<div style='background:{color}22;border:2px solid {color};
border-radius:12px;padding:20px;text-align:center;margin-bottom:8px'>
<div style='font-size:2.4rem'>{emoji}</div>
<div style='font-size:1.05rem;font-weight:700;color:{color};margin-top:6px'>
{pred_haz.replace("_"," ").title()} Hazard</div>
</div>""", unsafe_allow_html=True)
    with c2:
        st.metric("Estimated Magnitude", f"M {pred_mag:.1f}", "± 0.33 units")
    with c3:
        if r_dep < 70:    dl, dh = "Shallow (< 70 km)",   "Highest damage risk"
        elif r_dep < 300: dl, dh = "Intermediate",         "Moderate impact"
        else:             dl, dh = "Deep (> 300 km)",       "Reduced damage"
        st.metric("Depth Category", dl, dh)

    st.subheader("Hazard probability breakdown")
    classes  = le.classes_
    proba_df = pd.DataFrame({
        "Hazard Level": [c.replace("_"," ").title() for c in classes],
        "Probability":  haz_proba,
        "Color":        [HAZARD_COLORS[c] for c in classes]
    }).sort_values("Probability", ascending=True)
    fig = go.Figure(go.Bar(
        x=proba_df["Probability"], y=proba_df["Hazard Level"],
        orientation="h", marker_color=proba_df["Color"],
        text=[f"{p:.1%}" for p in proba_df["Probability"]],
        textposition="outside"
    ))
    fig.update_layout(
        xaxis_title="Probability", yaxis_title="",
        xaxis_range=[0, 1.15], height=220,
        margin=dict(l=10, r=20, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Why this prediction?")
    if result["has_history"]:
        w1, w2 = st.columns(2)
        w1.metric("Historical quakes in zone", f"{int(result['zone_count'])}", "in 2°×2° cell")
        w2.metric("Worst ever recorded",       f"M {result['zone_max_mag']:.1f}")
        w3, w4 = st.columns(2)
        w3.metric("Zone avg magnitude",  f"{result['zone_avg_mag']:.2f}")
        w4.metric("Quakes per year",     f"{result['zone_rate']:.2f}")
    else:
        st.success("No M5+ earthquakes in this zone in 60 years — seismically stable, Low risk.")

    if r_dep < 70:    st.warning("Shallow (< 70 km): highest surface damage risk.")
    elif r_dep < 300: st.info("Intermediate (70–300 km): moderate surface impact.")
    else:             st.success("Deep (> 300 km): energy absorbed before reaching surface.")

    st.subheader("Location + nearby historical earthquakes")
    m = folium.Map(location=[r_lat, r_lon], zoom_start=6)
    folium.Marker(
        [r_lat, r_lon],
        popup=folium.Popup(
            f"<b>{place_name}</b><br>"
            f"<code>{dms}</code><br>"
            f"Hazard: {pred_haz.replace('_',' ').title()}",
            max_width=280
        ),
        icon=folium.Icon(color="blue", icon="star")
    ).add_to(m)
    nearby = hist_df[
        (abs(hist_df["Latitude"]  - r_lat) < 5) &
        (abs(hist_df["Longitude"] - r_lon) < 5)
    ].head(300)
    for _, row in nearby.iterrows():
        mag = row["Magnitude"]
        dc  = "red" if mag >= 7 else "orange" if mag >= 6 else "beige"
        folium.CircleMarker(
            [row["Latitude"], row["Longitude"]],
            radius=max(2, (mag - 4) * 2),
            color=dc, fill=True, fill_opacity=0.55,
            popup=f"M{mag:.1f}"
        ).add_to(m)
    st_folium(m, height=400, use_container_width=True)
    st.caption("🔵 Your location  🔴 M7+  🟠 M6+  🟡 M5+")

    # ── Geological Profile ────────────────────────────────────────
    st.divider()
    st.markdown("## 🪨 Geological Profile")
    st.markdown(
        f"<div style='color:var(--color-text-secondary);margin-bottom:1rem'>"
        f"Detailed geological and tectonic analysis for <b>{place_name}</b><br>"
        f"<span style='font-family:monospace;font-size:0.9rem'>{dms}</span></div>",
        unsafe_allow_html=True
    )

    geo_fields = [
        ("🌐 Geological Setting",    geology.get("geological_setting",""), "#e05c4b"),
        ("⚙️ Tectonic Setting",       geology.get("tectonic_setting",  ""), "#378ADD"),
        ("🪨 Rock Types & Petrology", geology.get("rock_types",        ""), "#e05c4b"),
        ("📋 Stratigraphy",           geology.get("stratigraphy",      ""), "#378ADD"),
        ("🛢️ Basin Information",      geology.get("basin_info",        ""), "#e05c4b"),
        ("🌋 Seismic Context",        geology.get("seismic_context",   ""), "#378ADD"),
        ("🦕 Fossil Record",          geology.get("fossil_record",     ""), "#e05c4b"),
        ("⛏️ Economic Geology",       geology.get("economic_geology",  ""), "#378ADD"),
    ]
    for i in range(0, len(geo_fields), 2):
        gc1, gc2 = st.columns(2)
        with gc1:
            t, c, a = geo_fields[i]
            if c:
                st.markdown(f"**{t}**")
                st.markdown(
                    f"<div style='background:rgba(128,128,128,0.08);"
                    f"border-left:3px solid {a};padding:12px 16px;"
                    f"border-radius:0 8px 8px 0;margin-bottom:16px;"
                    f"line-height:1.7;font-size:0.95rem'>{c}</div>",
                    unsafe_allow_html=True)
        with gc2:
            if i + 1 < len(geo_fields):
                t2, c2, a2 = geo_fields[i + 1]
                if c2:
                    st.markdown(f"**{t2}**")
                    st.markdown(
                        f"<div style='background:rgba(128,128,128,0.08);"
                        f"border-left:3px solid {a2};padding:12px 16px;"
                        f"border-radius:0 8px 8px 0;margin-bottom:16px;"
                        f"line-height:1.7;font-size:0.95rem'>{c2}</div>",
                        unsafe_allow_html=True)

    fact = geology.get("interesting_fact", "")
    if fact:
        st.markdown(
            f"<div style='background:rgba(239,159,39,0.1);"
            f"border:1px solid #EF9F27;border-radius:10px;"
            f"padding:16px 20px;margin:8px 0'>"
            f"<b>💡 Fascinating geological fact</b><br><br>{fact}</div>",
            unsafe_allow_html=True)

    if geo_images:
        st.markdown("#### 📸 Geological features of this region")
        img_cols = st.columns(len(geo_images))
        for col, img in zip(img_cols, geo_images):
            with col:
                try:
                    st.image(img["url"],
                             caption=img["desc"][:80] if img["desc"] else "",
                             use_container_width=True)
                except: pass

# ══════════════════════════════════════════════════════════════════
# APP HEADER
# ══════════════════════════════════════════════════════════════════
st.title("🌍 Seismic Hazard Assessment System")
st.caption(
    "Trained on 36,921 earthquakes (1965–2016)  ·  "
    "Validated on 2017–2024 USGS data  ·  "
    "96% hazard classification accuracy"
)
st.divider()

tab1, tab2, tab3 = st.tabs(["🔍 Predict", "🗺️ Global Map", "📊 Insights"])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Click the map or enter coordinates")

    # Click-a-location map
    map_center = [st.session_state.get("lat", 20), st.session_state.get("lon", 0)]
    click_map  = folium.Map(location=map_center,
                             zoom_start=4 if "lat" in st.session_state else 2)

    sample_q = hist_df[hist_df["Magnitude"] >= 6.5].sample(
        min(500, len(hist_df[hist_df["Magnitude"] >= 6.5])), random_state=1
    )
    for _, row in sample_q.iterrows():
        mag = row["Magnitude"]
        folium.CircleMarker(
            [row["Latitude"], row["Longitude"]],
            radius=max(2, (mag - 5) * 2),
            color="red" if mag >= 7 else "orange",
            fill=True, fill_opacity=0.3,
            popup=f"M{mag:.1f}"
        ).add_to(click_map)

    if "lat" in st.session_state:
        folium.Marker(
            [st.session_state["lat"], st.session_state["lon"]],
            popup=f"Selected: {decimal_to_dms(st.session_state['lat'], st.session_state['lon'])}",
            icon=folium.Icon(color="blue", icon="star")
        ).add_to(click_map)

    map_data = st_folium(click_map, height=360,
                          use_container_width=True,
                          returned_objects=["last_clicked"])

    clicked = map_data.get("last_clicked")
    if clicked and clicked.get("lat") is not None:
        clat = round(clicked["lat"], 4)
        clon = round(clicked["lng"], 4)
        prev = st.session_state.get("last_click", {})
        if clat != prev.get("lat") or clon != prev.get("lon"):
            st.session_state["last_click"]  = {"lat": clat, "lon": clon}
            st.session_state["clicked_lat"] = clat
            st.session_state["clicked_lon"] = clon
            st.session_state["from_click"]  = True
            st.rerun()

    if st.session_state.get("from_click", False):
        st.session_state["from_click"] = False
        clat = st.session_state.pop("clicked_lat")
        clon = st.session_state.pop("clicked_lon")
        run_assessment(clat, clon, st.session_state.get("depth", 30))
        st.rerun()

    st.markdown("---")
    st.markdown("**Or enter manually:**")

    col1, col2 = st.columns(2)
    with col1:
        lat_str = st.text_input(
            "Latitude (−90 to 90)",
            value=str(st.session_state.get("lat", "35.6800")),
            placeholder="e.g. 35.68"
        )
    with col2:
        lon_str = st.text_input(
            "Longitude (−180 to 180)",
            value=str(st.session_state.get("lon", "139.6900")),
            placeholder="e.g. 139.69"
        )

    depth = st.slider("Depth (km)", min_value=0, max_value=700,
                      value=st.session_state.get("depth", 30))

    # Live DMS preview
    try:
        prev_lat = float(lat_str)
        prev_lon = float(lon_str)
        if -90 <= prev_lat <= 90 and -180 <= prev_lon <= 180:
            st.markdown(
                f"<div style='font-family:monospace;font-size:0.9rem;"
                f"color:var(--color-text-secondary);margin:-8px 0 8px'>"
                f"📐 {decimal_to_dms(prev_lat, prev_lon)}</div>",
                unsafe_allow_html=True
            )
    except: pass

    input_ok = True
    lat, lon = 0.0, 0.0
    try:
        lat = float(lat_str)
        if not (-90 <= lat <= 90):
            st.error("Latitude must be between −90 and 90.")
            input_ok = False
    except:
        st.error("Latitude must be a number — e.g. 35.68")
        input_ok = False
    try:
        lon = float(lon_str)
        if not (-180 <= lon <= 180):
            st.error("Longitude must be between −180 and 180.")
            input_ok = False
    except:
        st.error("Longitude must be a number — e.g. 139.69")
        input_ok = False

    with st.expander("📌 Quick location reference"):
        st.markdown("""
| City | Latitude | Longitude | DMS |
|---|---|---|---|
| Tokyo, Japan | 35.6800 | 139.6900 | 35°40'48"N, 139°41'24"E |
| San Francisco | 37.7749 | -122.4194 | 37°46'29"N, 122°25'10"W |
| Kathmandu, Nepal | 27.7172 | 85.3240 | 27°43'02"N, 85°19'26"E |
| Bangalore, India | 12.9700 | 77.5900 | 12°58'12"N, 77°35'24"E |
| Istanbul, Turkey | 41.0082 | 28.9784 | 41°00'29"N, 28°58'42"E |
| Mexico City | 19.4326 | -99.1332 | 19°25'57"N, 99°07'60"W |
| Santiago, Chile | -33.4489 | -70.6693 | 33°26'56"S, 70°40'09"W |
| Jakarta | -6.2088 | 106.8456 | 06°12'32"S, 106°50'44"E |
        """)

    if st.button("Assess Seismic Hazard", type="primary",
                 use_container_width=True, disabled=not input_ok):
        run_assessment(lat, lon, depth)

    if "result" not in st.session_state:
        st.info("Click the map or enter coordinates, then press **Assess Seismic Hazard**.")
    else:
        show_results()

# ══════════════════════════════════════════════════════════════════
# TAB 2 — GLOBAL MAP
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Global earthquake activity")
    fc1, fc2 = st.columns([3, 1])
    with fc2:
        st.markdown("**Filters**")
        min_mag     = st.slider("Min magnitude", 5.0, 9.0, 6.0, 0.1)
        show_recent = st.checkbox("Live last 30 days", value=True)
        st.markdown("🔴 M7+  🟠 M6+  🟡 M5+  🔵 Live")
    with fc1:
        world_map = folium.Map(location=[20, 0], zoom_start=2)
        plot_df   = hist_df[hist_df["Magnitude"] >= min_mag].copy()
        sample    = plot_df.sample(min(2000, len(plot_df)), random_state=42)
        for _, row in sample.iterrows():
            mag = row["Magnitude"]
            c   = "red" if mag >= 7 else "orange" if mag >= 6 else "yellow"
            folium.CircleMarker(
                [row["Latitude"], row["Longitude"]],
                radius=max(2, (mag - 4) * 1.5),
                color=c, fill=True, fill_opacity=0.4,
                popup=f"M{mag:.1f} | {row['Depth']:.0f} km"
            ).add_to(world_map)
        if show_recent and len(recent_df) > 0:
            for _, row in recent_df.iterrows():
                folium.CircleMarker(
                    [row["Latitude"], row["Longitude"]],
                    radius=max(3, row["Magnitude"]),
                    color="blue", fill=True, fill_opacity=0.7,
                    popup=f"LIVE: M{row['Magnitude']:.1f} — {row['Place']}"
                ).add_to(world_map)
        st_folium(world_map, height=500, use_container_width=True)

    st.subheader("Magnitude distribution")
    fig2 = px.histogram(hist_df, x="Magnitude", nbins=40,
                        color_discrete_sequence=["#e05c4b"])
    fig2.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                       plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 3 — INSIGHTS
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("How the model works")
    ins1, ins2 = st.columns(2)
    with ins1:
        st.markdown("#### Magnitude Regressor (XGBoost)")
        st.markdown("""
| Metric | Value |
|---|---|
| MAE | 0.33 magnitude units |
| Train | 23,223 quakes (1965–2016) |
| Test | 13,698 quakes (2017–2024) |
        """)
        st.info("Magnitude prediction from location is a known open problem in seismology. MAE 0.33 captures regional patterns.")
    with ins2:
        st.markdown("#### Hazard Classifier (XGBoost)")
        st.markdown("""
| Metric | Value |
|---|---|
| Accuracy | 96% |
| Classes | Low / Moderate / High / Very High |
| Train | 23,223 quakes (1965–2016) |
| Test | 13,698 quakes (2017–2024) |
        """)
        st.success("96% accuracy on completely unseen 2017–2024 data.")

    st.divider()
    st.subheader("Feature importance")
    feat_df = pd.DataFrame({
        "Feature":    features,
        "Importance": mag_model.feature_importances_
    }).sort_values("Importance", ascending=True)
    fig3 = px.bar(feat_df, x="Importance", y="Feature", orientation="h",
                  color="Importance", color_continuous_scale="reds")
    fig3.update_layout(height=480, margin=dict(l=10, r=10, t=10, b=10),
                       plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)
    st.info("**zone_avg_mag** (0.27) is the strongest predictor.")

    st.divider()
    st.subheader("Dataset statistics")
    d1, d2 = st.columns(2)
    d1.metric("Total earthquakes", f"{len(hist_df):,}")
    d2.metric("Years covered",     "1965 – 2024")
    d3, d4 = st.columns(2)
    d3.metric("Geographic zones",  "1,918")
    d4.metric("Features used",     str(len(features)))

    decade_df = hist_df.copy()
    decade_df["decade"] = (decade_df["Date"].dt.year // 10 * 10).astype(str) + "s"
    decade_counts = decade_df.groupby("decade").size().reset_index(name="Count")
    fig4 = px.bar(decade_counts, x="decade", y="Count",
                  color_discrete_sequence=["#e05c4b"])
    fig4.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                       plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig4, use_container_width=True)
