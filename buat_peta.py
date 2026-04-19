"""
Script untuk membuat visualisasi peta interaktif tingkat kriminalitas Jawa Timur.
Menggunakan data dari file Excel dan GeoJSON Kabupaten/Kota.
Output: peta_kriminalitas_jatim.html
"""

import glob
import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. KONFIGURASI
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR  = "tingkat kriminalitas"
GEOJSON   = "Kabupaten-Kota (Provinsi Jawa Timur).geojson"
OUT_HTML  = "peta_kriminalitas_jatim.html"
RANDOM_STATE = 42

COL_NAMES    = ["Kabupaten_Kota", "Jumlah_Kejahatan", "Risiko_100k",
                "Persen_Penyelesaian", "Selang_Waktu"]
FEATURE_COLS = COL_NAMES[1:]

# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_year_data(filepath: str) -> pd.DataFrame:
    year = int(os.path.basename(filepath).split()[-1].replace(".xlsx", ""))
    df   = pd.read_excel(filepath, header=0)
    df.columns = COL_NAMES

    mask_jatim = df["Kabupaten_Kota"].astype(str).str.lower().str.contains("jawa timur", na=True)
    df = df[~mask_jatim].copy()

    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["Kabupaten_Kota"].notna()].copy()
    df = df[~df["Kabupaten_Kota"].astype(str).str.match(
        r"^[<\d]|catatan|sumber|source|note", flags=0, na=True
    )].copy()

    valid_mask = df[FEATURE_COLS].notna().sum(axis=1) >= 3
    df = df[valid_mask].copy()
    df["Tahun"] = year
    return df


FILES = sorted(glob.glob(os.path.join(DATA_DIR, "*.xlsx")))
dfs   = [load_year_data(f) for f in FILES]
df_raw = pd.concat(dfs, ignore_index=True)

from scipy.stats import linregress
def calc_slope(df_group):
    df_sorted = df_group.sort_values("Tahun")
    y = df_sorted["Risiko_100k"].values
    if len(y) > 1:
        return linregress(np.arange(len(y)), y).slope
    return 0.0

slope_df = df_raw.groupby("Kabupaten_Kota").apply(calc_slope).reset_index()
slope_df.columns = ["Kabupaten_Kota", "Slope_Risiko"]

df_profile = (df_raw
              .groupby("Kabupaten_Kota", sort=True)[FEATURE_COLS]
              .median()
              .reset_index())
df_profile = df_profile.merge(slope_df, on="Kabupaten_Kota", how="left")

print(f"✅ Data loaded: {df_profile.shape[0]} kabupaten/kota")

# ─────────────────────────────────────────────────────────────────────────────
# 3. OUTLIER HANDLING (IQR Winsorization)
# ─────────────────────────────────────────────────────────────────────────────

def detect_and_cap_iqr(df, cols):
    df_out = df.copy()
    for col in cols:
        q1  = df[col].quantile(0.25)
        q3  = df[col].quantile(0.75)
        iqr = q3 - q1
        lo  = q1 - 1.5 * iqr
        hi  = q3 + 1.5 * iqr
        df_out[col] = df_out[col].clip(lower=lo, upper=hi)
    return df_out


df_clean = detect_and_cap_iqr(df_profile, FEATURE_COLS)

# ─────────────────────────────────────────────────────────────────────────────
# 4. STANDARDISASI
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# 5. HARDCODED CLUSTER ASSIGNMENT DARI MODEL.IPYNB
#    Sumber: K-Means K=4 + Danger Score komposit di model.ipynb
#    Di-hardcode agar konsisten — tidak re-run clustering sendiri
# ─────────────────────────────────────────────────────────────────────────────

# Format: "Nama Kab/Kota" → ("Risk Label", "#HexColor", "🎨Emoji", Danger_Score)
CLUSTER_MAP = {
    # 🔴 High Risk — Klaster 2, Danger Score ~47
    "Banyuwangi"       : ("High Risk",          "#e74c3c", "🔴", 47.08),
    "Bojonegoro"       : ("High Risk",          "#e74c3c", "🔴", 47.08),
    "Gresik"           : ("High Risk",          "#e74c3c", "🔴", 47.08),
    "Jember"           : ("High Risk",          "#e74c3c", "🔴", 47.08),
    "Jombang"          : ("High Risk",          "#e74c3c", "🔴", 47.08),
    "Kediri"           : ("High Risk",          "#e74c3c", "🔴", 47.08),
    "Kota Surabaya"    : ("High Risk",          "#e74c3c", "🔴", 47.08),
    "Lamongan"         : ("High Risk",          "#e74c3c", "🔴", 47.08),
    "Malang"           : ("High Risk",          "#e74c3c", "🔴", 47.08),
    "Pasuruan"         : ("High Risk",          "#e74c3c", "🔴", 47.08),
    "Sidoarjo"         : ("High Risk",          "#e74c3c", "🔴", 47.08),
    "Tulungagung"      : ("High Risk",          "#e74c3c", "🔴", 47.08),
    # 🟠 Moderate-High Risk — Klaster 4, Danger Score ~38
    "Kota Kediri"      : ("Moderate-High Risk", "#e67e22", "🟠", 38.45),
    "Kota Mojokerto"   : ("Moderate-High Risk", "#e67e22", "🟠", 38.45),
    "Kota Pasuruan"    : ("Moderate-High Risk", "#e67e22", "🟠", 38.45),
    "Kota Probolinggo" : ("Moderate-High Risk", "#e67e22", "🟠", 38.45),
    "Magetan"          : ("Moderate-High Risk", "#e67e22", "🟠", 38.45),
    "Ngawi"            : ("Moderate-High Risk", "#e67e22", "🟠", 38.45),
    # 🟡 Moderate-Low Risk — Klaster 3, Danger Score ~35
    "Bondowoso"        : ("Moderate-Low Risk",  "#f1c40f", "🟡", 34.62),
    "Kota Malang"      : ("Moderate-Low Risk",  "#f1c40f", "🟡", 34.62),
    "Lumajang"         : ("Moderate-Low Risk",  "#f1c40f", "🟡", 34.62),
    "Mojokerto"        : ("Moderate-Low Risk",  "#f1c40f", "🟡", 34.62),
    "Pamekasan"        : ("Moderate-Low Risk",  "#f1c40f", "🟡", 34.62),
    "Situbondo"        : ("Moderate-Low Risk",  "#f1c40f", "🟡", 34.62),
    "Sumenep"          : ("Moderate-Low Risk",  "#f1c40f", "🟡", 34.62),
    "Tuban"            : ("Moderate-Low Risk",  "#f1c40f", "🟡", 34.62),
    # 🟢 Safe Zone — Klaster 1, Danger Score ~23
    "Bangkalan"        : ("Safe Zone",          "#2ecc71", "🟢", 23.14),
    "Blitar"           : ("Safe Zone",          "#2ecc71", "🟢", 23.14),
    "Kota Batu"        : ("Safe Zone",          "#2ecc71", "🟢", 23.14),
    "Kota Blitar"      : ("Safe Zone",          "#2ecc71", "🟢", 23.14),
    "Kota Madiun"      : ("Safe Zone",          "#2ecc71", "🟢", 23.14),
    "Madiun"           : ("Safe Zone",          "#2ecc71", "🟢", 23.14),
    "Nganjuk"          : ("Safe Zone",          "#2ecc71", "🟢", 23.14),
    "Pacitan"          : ("Safe Zone",          "#2ecc71", "🟢", 23.14),
    "Ponorogo"         : ("Safe Zone",          "#2ecc71", "🟢", 23.14),
    "Probolinggo"      : ("Safe Zone",          "#2ecc71", "🟢", 23.14),
    "Sampang"          : ("Safe Zone",          "#2ecc71", "🟢", 23.14),
    "Trenggalek"       : ("Safe Zone",          "#2ecc71", "🟢", 23.14),
}

# Terapkan ke df_clean
df_clean["Risk_Label"]   = df_clean["Kabupaten_Kota"].map(lambda n: CLUSTER_MAP.get(n, ("No Data","#AAAAAA","⬜",0))[0])
df_clean["Risk_Color"]   = df_clean["Kabupaten_Kota"].map(lambda n: CLUSTER_MAP.get(n, ("No Data","#AAAAAA","⬜",0))[1])
df_clean["Risk_Emoji"]   = df_clean["Kabupaten_Kota"].map(lambda n: CLUSTER_MAP.get(n, ("No Data","#AAAAAA","⬜",0))[2])
df_clean["Danger_Score"] = df_clean["Kabupaten_Kota"].map(lambda n: CLUSTER_MAP.get(n, ("No Data","#AAAAAA","⬜",0))[3])

# Kolom Kasus Tidak Selesai (untuk popup)
df_clean["Kasus_Tidak_Selesai"] = (
    df_clean["Jumlah_Kejahatan"] * (1 - df_clean["Persen_Penyelesaian"] / 100)
).round(1)

print("✅ Cluster assignment dari model.ipynb diterapkan:")
for label in ["High Risk", "Moderate-High Risk", "Moderate-Low Risk", "Safe Zone"]:
    members = sorted(df_clean.loc[df_clean["Risk_Label"] == label, "Kabupaten_Kota"].tolist())
    emoji   = CLUSTER_MAP[members[0]][2] if members else ""
    danger  = CLUSTER_MAP[members[0]][3] if members else 0
    print(f"   {emoji} {label} ({len(members)} daerah) | Danger={danger}")
    print(f"      {', '.join(members)}")

# Buat cluster_to_risk & centroid_tbl_s agar blok downstream tetap kompatibel
_label_order = ["High Risk", "Moderate-High Risk", "Moderate-Low Risk", "Safe Zone"]
_colors      = {"High Risk": "#e74c3c", "Moderate-High Risk": "#e67e22",
                "Moderate-Low Risk": "#f1c40f", "Safe Zone": "#2ecc71"}
_emojis      = {"High Risk": "🔴", "Moderate-High Risk": "🟠",
                "Moderate-Low Risk": "🟡", "Safe Zone": "🟢"}
_dangers     = {"High Risk": 47.08, "Moderate-High Risk": 38.45,
                "Moderate-Low Risk": 34.62, "Safe Zone": 23.14}

cluster_to_risk = {}
for i, lbl in enumerate(_label_order):
    grp = df_clean[df_clean["Risk_Label"] == lbl]
    cluster_to_risk[i] = {
        "label" : lbl,
        "color" : _colors[lbl],
        "emoji" : _emojis[lbl],
        "danger": _dangers[lbl],
        "n"     : len(grp),
        "stats" : {
            "Jml Kejahatan"    : round(grp["Jumlah_Kejahatan"].mean(),    1),
            "Kasus Tdk Selesai": round(grp["Kasus_Tidak_Selesai"].mean(), 1),
            "Risiko/100k"      : round(grp["Risiko_100k"].mean(),         1),
            "% Penyelesaian"   : round(grp["Persen_Penyelesaian"].mean(), 2),
            "Selang Waktu(j)"  : round(grp["Selang_Waktu"].mean(),        1),
        }
    }
    # Tambah kolom Cluster_KMeans palsu (virtual) agar lookup tidak error
    df_clean.loc[df_clean["Risk_Label"] == lbl, "Cluster_KMeans"] = i

# ─────────────────────────────────────────────────────────────────────────────
# 7. MAPPING: Data ↔ GeoJSON
# ─────────────────────────────────────────────────────────────────────────────

NAME_MAP = {
    "Kota Batu"     : "Batu",
    "Kota Surabaya" : "Surabaya",
}

df_clean["GeoName"] = df_clean["Kabupaten_Kota"].replace(NAME_MAP)

threshold_bahaya = max(df_clean["Slope_Risiko"].quantile(0.75), 0.1)
df_clean["Hidden_Risk"] = (df_clean["Risk_Label"].str.contains("Safe|Low")) & (df_clean["Slope_Risiko"] > threshold_bahaya)

with open(GEOJSON, "r", encoding="utf-8") as f:
    geo = json.load(f)

geo_names = {feat["properties"]["NAME_2"] for feat in geo["features"]}
data_names = set(df_clean["GeoName"])
unmatched = data_names - geo_names
if unmatched:
    print(f"⚠️  Unmatched names: {unmatched}")
else:
    print("✅ All region names matched")

# ─────────────────────────────────────────────────────────────────────────────
# 8. BUILD GeoJSON enriched dengan data
# ─────────────────────────────────────────────────────────────────────────────

lookup = df_clean.set_index("GeoName")

feature_list = []
for feat in geo["features"]:
    gname = feat["properties"]["NAME_2"]
    if gname in lookup.index:
        row  = lookup.loc[gname]
        cid  = int(row["Cluster_KMeans"])
        info = cluster_to_risk[cid]
        props = {
            "name"            : gname,
            "cluster"         : cid + 1,
            "risk_label"      : info["label"],
            "risk_color"      : info["color"],
            "risk_emoji"      : info["emoji"],
            "danger_score"    : info["danger"],
            "jumlah_kejahatan": float(row["Jumlah_Kejahatan"]),
            "risiko_100k"     : float(row["Risiko_100k"]),
            "persen_selesai"  : float(row["Persen_Penyelesaian"]),
            "selang_waktu"    : float(row["Selang_Waktu"]),
            "slope_risiko"    : float(row["Slope_Risiko"]),
            "hidden_risk"     : bool(row["Hidden_Risk"]),
        }
    else:
        props = {
            "name"        : gname,
            "cluster"     : None,
            "risk_label"  : "No Data",
            "risk_color"  : "#cccccc",
            "risk_emoji"  : "⬜",
            "danger_score": None,
            "jumlah_kejahatan": None,
            "risiko_100k" : None,
            "persen_selesai": None,
            "selang_waktu": None,
            "slope_risiko"    : None,
            "hidden_risk"     : False,
        }
    feature_list.append({
        "type"      : "Feature",
        "properties": props,
        "geometry"  : feat["geometry"],
    })

enriched_geo = {
    "type"    : "FeatureCollection",
    "features": feature_list,
}

# ─────────────────────────────────────────────────────────────────────────────
# 9. Siapkan data legenda & statistik ringkasan untuk JS
# ─────────────────────────────────────────────────────────────────────────────

legend_items = []
for i, lbl in enumerate(_label_order):
    info    = cluster_to_risk[i]
    members = sorted(df_clean.loc[df_clean["Risk_Label"] == lbl, "Kabupaten_Kota"].tolist())
    legend_items.append({
        "label"  : lbl,
        "color"  : info["color"],
        "emoji"  : info["emoji"],
        "danger" : info["danger"],
        "n"      : info["n"],
        "members": members,
        "stats"  : info["stats"],
    })

# ─────────────────────────────────────────────────────────────────────────────
# 10. GENERATE HTML
# ─────────────────────────────────────────────────────────────────────────────

geo_json_str    = json.dumps(enriched_geo, ensure_ascii=False)
legend_json_str = json.dumps(legend_items, ensure_ascii=False)

html = f"""<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Peta Tingkat Kriminalitas Jawa Timur</title>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet"/>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    html {{ height: 100%; }}

    :root {{
      --bg-dark    : #0f1117;
      --bg-card    : #1a1d27;
      --bg-card2   : #21253a;
      --accent     : #6c8ef5;
      --accent2    : #a78bfa;
      --text-main  : #e8eaf0;
      --text-sub   : #9ba3c4;
      --border     : rgba(108,142,245,0.18);
      --red        : #e74c3c;
      --orange     : #e67e22;
      --yellow     : #f1c40f;
      --green      : #2ecc71;
      --radius     : 14px;
      --shadow     : 0 8px 32px rgba(0,0,0,0.5);
    }}

    body {{
      font-family: 'Inter', sans-serif;
      background: var(--bg-dark);
      color: var(--text-main);
      height: 100vh;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }}

    /* ── HEADER ───────────────────────── */
    header {{
      background: linear-gradient(135deg, #12143a 0%, #1b1f3a 50%, #0d2044 100%);
      border-bottom: 1px solid var(--border);
      padding: 22px 32px 18px;
      display: flex;
      align-items: center;
      gap: 18px;
    }}
    .header-icon {{
      font-size: 2.4rem;
      filter: drop-shadow(0 0 12px rgba(108,142,245,0.5));
    }}
    .header-text h1 {{
      font-size: 1.45rem;
      font-weight: 700;
      background: linear-gradient(90deg, #818cf8, #c084fc);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      line-height: 1.2;
    }}
    .header-text p {{
      font-size: 0.82rem;
      color: var(--text-sub);
      margin-top: 4px;
    }}

    /* ── MAIN LAYOUT ──────────────────── */
    .main {{
      display: flex;
      flex: 1;
      min-height: 0;
      overflow: hidden;
    }}

    /* ── SIDEBAR ──────────────────────── */
    .sidebar {{
      width: 330px;
      min-width: 290px;
      background: var(--bg-card);
      border-right: 1px solid var(--border);
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      padding: 20px 16px;
      gap: 18px;
    }}

    .section-title {{
      font-size: 0.72rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--accent);
      margin-bottom: 10px;
    }}

    /* Legend Cards */
    .legend-card {{
      background: var(--bg-card2);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 14px 16px;
      cursor: pointer;
      transition: transform 0.18s, box-shadow 0.18s, border-color 0.18s;
    }}
    .legend-card:hover, .legend-card.active {{
      transform: translateY(-2px);
      box-shadow: var(--shadow);
      border-color: var(--accent);
    }}
    .legend-card-header {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 10px;
    }}
    .risk-dot {{
      width: 14px;
      height: 14px;
      border-radius: 50%;
      flex-shrink: 0;
      box-shadow: 0 0 8px currentColor;
    }}
    .risk-name {{
      font-size: 0.88rem;
      font-weight: 600;
      flex: 1;
    }}
    .risk-badge {{
      font-size: 0.7rem;
      font-weight: 600;
      padding: 2px 8px;
      border-radius: 20px;
      background: rgba(255,255,255,0.08);
    }}
    .stat-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
    }}
    .stat-item {{
      background: rgba(0,0,0,0.25);
      border-radius: 8px;
      padding: 6px 8px;
    }}
    .stat-label {{
      font-size: 0.65rem;
      color: var(--text-sub);
      margin-bottom: 2px;
    }}
    .stat-val {{
      font-size: 0.82rem;
      font-weight: 600;
    }}
    .danger-bar-wrap {{
      margin-top: 10px;
    }}
    .danger-bar-label {{
      font-size: 0.65rem;
      color: var(--text-sub);
      display: flex;
      justify-content: space-between;
      margin-bottom: 4px;
    }}
    .danger-bar-bg {{
      background: rgba(255,255,255,0.08);
      border-radius: 4px;
      height: 6px;
      overflow: hidden;
    }}
    .danger-bar-fill {{
      height: 100%;
      border-radius: 4px;
      transition: width 0.6s ease;
    }}
    .members-toggle {{
      margin-top: 10px;
      font-size: 0.72rem;
      color: var(--accent);
      cursor: pointer;
      user-select: none;
    }}
    .members-list {{
      display: none;
      margin-top: 6px;
      font-size: 0.73rem;
      color: var(--text-sub);
      line-height: 1.7;
    }}
    .members-list.open {{ display: block; }}

    /* Info Panel (click popup) */
    .info-panel {{
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 14px 16px;
    }}
    .info-panel .ip-name {{
      font-size: 1rem;
      font-weight: 700;
      margin-bottom: 10px;
    }}
    .info-panel .ip-row {{
      display: flex;
      justify-content: space-between;
      font-size: 0.8rem;
      padding: 4px 0;
      border-bottom: 1px solid rgba(255,255,255,0.05);
    }}
    .info-panel .ip-row:last-child {{ border: none; }}
    .info-panel .ip-key   {{ color: var(--text-sub); }}
    .info-panel .ip-value {{ font-weight: 600; }}
    .info-placeholder {{
      font-size: 0.8rem;
      color: var(--text-sub);
      text-align: center;
      padding: 12px 0;
    }}

    /* ── MAP ──────────────────────────── */
    #map {{
      flex: 1;
      min-height: 0;
      height: 100%;
      background: #0d0f1a;
    }}

    /* ── LEAFLET POPUP override ─────── */
    .leaflet-popup-content-wrapper {{
      background: var(--bg-card) !important;
      border: 1px solid var(--border) !important;
      border-radius: 10px !important;
      box-shadow: var(--shadow) !important;
      color: var(--text-main) !important;
      font-family: 'Inter', sans-serif !important;
      font-size: 0.82rem !important;
    }}
    .leaflet-popup-tip {{
      background: var(--bg-card) !important;
    }}
    .leaflet-popup-content {{
      margin: 12px 14px !important;
      line-height: 1.6 !important;
    }}
    .popup-title {{
      font-size: 0.95rem;
      font-weight: 700;
      margin-bottom: 8px;
    }}
    .popup-risk {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 3px 10px;
      border-radius: 20px;
      font-size: 0.72rem;
      font-weight: 600;
      margin-bottom: 10px;
    }}
    .popup-row {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 3px 0;
      border-bottom: 1px solid rgba(255,255,255,0.06);
    }}
    .popup-row:last-child {{ border: none; }}
    .popup-key  {{ color: #9ba3c4; }}
    .popup-val  {{ font-weight: 600; }}

    /* ── FOOTER ───────────────────────── */
    footer {{
      text-align: center;
      font-size: 0.72rem;
      color: var(--text-sub);
      padding: 10px 20px;
      border-top: 1px solid var(--border);
      background: var(--bg-dark);
    }}

    /* scrollbar */
    ::-webkit-scrollbar {{ width: 6px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}

    /* ── STRATEGY PANEL ───────────────────── */
    .strategy-panel {{
      background: var(--bg-card2);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 14px 16px;
      margin-top: 12px;
    }}
    .strategy-header {{
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 10px;
    }}
    .strategy-icon {{
      font-size: 1.1rem;
    }}
    .strategy-title {{
      font-size: 0.8rem;
      font-weight: 700;
      color: var(--text-main);
    }}
    .strategy-subtitle {{
      font-size: 0.68rem;
      color: var(--text-sub);
      margin-bottom: 10px;
      font-style: italic;
    }}
    .strategy-list {{
      list-style: none;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .strategy-list li {{
      display: flex;
      align-items: flex-start;
      gap: 8px;
      font-size: 0.76rem;
      color: var(--text-main);
      line-height: 1.45;
      padding: 6px 8px;
      background: rgba(0,0,0,0.2);
      border-radius: 8px;
      border-left: 3px solid;
    }}
    .strategy-list li .s-num {{
      font-size: 0.65rem;
      font-weight: 700;
      min-width: 18px;
      height: 18px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
      margin-top: 1px;
    }}
    /* mini strategy in legend card */
    .card-strategy {{
      margin-top: 10px;
      padding: 8px 10px;
      background: rgba(0,0,0,0.2);
      border-radius: 8px;
      font-size: 0.7rem;
      color: var(--text-sub);
      border-left: 3px solid;
      display: none;
    }}
    .legend-card.active .card-strategy {{
      display: block;
    }}
    .card-strategy strong {{
      display: block;
      margin-bottom: 4px;
      font-size: 0.72rem;
    }}

    @media (max-width: 760px) {{
      .main {{ flex-direction: column; }}
      .sidebar {{ width: 100%; border-right: none; border-bottom: 1px solid var(--border); max-height: 340px; }}
    }}
  </style>
</head>
<body>

<header>
  <div class="header-icon">🗺️</div>
  <div class="header-text">
    <h1>Peta Kriminalitas Kabupaten/Kota Jawa Timur</h1>
    <p>Analisis Klaster K-Means berdasarkan data 2020–2024 &nbsp;|&nbsp; 38 Kabupaten/Kota &nbsp;|&nbsp; Sumber: BPS Jatim</p>
  </div>
</header>

<div class="main">
  <!-- SIDEBAR -->
  <aside class="sidebar" id="sidebar">
    <div>
      <div class="section-title">Profil Risiko Cluster</div>
      <div id="legend-cards"></div>
    </div>
    <div>
      <div class="section-title">Detail Daerah</div>
      <div id="info-panel">
        <div class="info-placeholder">Klik daerah pada peta untuk melihat detail.</div>
      </div>
    </div>
  </aside>

  <!-- MAP -->
  <div id="map"></div>
</div>

<footer>
  Visualisasi Peta Tingkat Kriminalitas Provinsi Jawa Timur &nbsp;·&nbsp;
  Data: Tingkat Kriminalitas 2020–2024 &nbsp;·&nbsp;
  Metode: K-Means Clustering (K=4) &nbsp;·&nbsp;
  Basemap: © OpenStreetMap contributors
</footer>

<script>
// ─── DATA ────────────────────────────────────────────────────────────────────
const GEO = {geo_json_str};
const LEGEND = {legend_json_str};

// ─── SARAN STRATEGIS PER LEVEL RISIKO ────────────────────────────────────────
const STRATEGIES = {{
  "High Risk": {{
    icon: "🚨",
    title: "Intervensi Segera Diperlukan",
    subtitle: "Tingkat kriminalitas dan kasus tak terselesaikan sangat tinggi. Diperlukan langkah prioritas lintas sektor.",
    color: "#e74c3c",
    actions: [
      "Tambah personel & pos keamanan di titik-titik rawan",
      "Bentuk satgas terpadu lintas dinas (Polri, Dinsos, Satpol PP)",
      "Prioritaskan penyelesaian kasus yang masih menunggak",
      "Aktifkan sistem pelaporan masyarakat berbasis digital",
      "Audit berkala kinerja penegak hukum & evaluasi SOP",
    ]
  }},
  "Moderate-High Risk": {{
    icon: "⚠️",
    title: "Penguatan Sistem Keamanan",
    subtitle: "Risiko signifikan. Diperlukan optimalisasi kapasitas penegakan hukum dan pencegahan berbasis data.",
    color: "#e67e22",
    actions: [
      "Optimalkan jadwal & rute patroli rutin di zona rawan",
      "Tingkatkan rasio penyelesaian kasus melalui pelatihan",
      "Analisis pola kejahatan untuk pencegahan proaktif",
      "Program pemberdayaan sosial-ekonomi cegah faktor kejahatan",
      "Perkuat koordinasi antar instansi keamanan daerah",
    ]
  }},
  "Moderate-Low Risk": {{
    icon: "📋",
    title: "Pemeliharaan & Pencegahan",
    subtitle: "Kondisi relatif aman, namun perlu kewaspadaan agar tidak meningkat. Fokus pada pemeliharaan dan edukasi.",
    color: "#f1c40f",
    actions: [
      "Pertahankan efektivitas waktu respons aparat",
      "Perkuat program penyuluhan hukum kepada warga",
      "Monitoring tren kriminalitas secara berkala",
      "Dorong partisipasi komunitas dalam keamanan lingkungan",
      "Tingkatkan literasi keamanan digital masyarakat",
    ]
  }},
  "Safe Zone": {{
    icon: "✅",
    title: "Pertahankan & Replikasi",
    subtitle: "Daerah dengan tingkat kriminalitas rendah dan penyelesaian baik. Jadikan model bagi daerah lain.",
    color: "#2ecc71",
    actions: [
      "Dokumentasikan praktik baik untuk direplikasi daerah lain",
      "Pertahankan rasio personel & anggaran keamanan saat ini",
      "Fokus pada program pencegahan dini berbasis komunitas",
      "Kembangkan program edukasi keamanan di sekolah",
      "Jadikan benchmark & mentor bagi kabupaten/kota sekitar",
    ]
  }},
}};

// ─── LEAFLET MAP ─────────────────────────────────────────────────────────────
const map = L.map('map', {{
  center: [-7.5, 112.3],
  zoom: 8,
  zoomControl: true,
  attributionControl: true,
}});

// Dark tile layer
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
  attribution: '© OpenStreetMap contributors &amp; CartoDB',
  subdomains: 'abcd',
  maxZoom: 19,
}}).addTo(map);

// ─── STYLE FUNCTIONS ─────────────────────────────────────────────────────────
let activeLayer = null;
let highlightedRisk = null;

function styleDefault(feature) {{
  return {{
    fillColor  : feature.properties.risk_color || '#555',
    fillOpacity: highlightedRisk
                   ? (feature.properties.risk_label === highlightedRisk ? 0.82 : 0.28)
                   : 0.7,
    color      : 'rgba(255,255,255,0.15)',
    weight     : 1,
    dashArray  : '',
  }};
}}

function styleHighlight(feature) {{
  return {{
    fillColor  : feature.properties.risk_color || '#555',
    fillOpacity: 0.92,
    color      : '#ffffff',
    weight     : 2.5,
    dashArray  : '',
  }};
}}

// ─── POPUP ───────────────────────────────────────────────────────────────────
function makePopup(p) {{
  const ds = p.danger_score !== null ? p.danger_score.toFixed(1) : '—';
  return `
    <div class="popup-title">${{p.name}} ${{p.hidden_risk ? '⚠️' : ''}}</div>
    <div class="popup-risk" style="background:${{p.risk_color}}22; color:${{p.risk_color}}; border:1px solid ${{p.risk_color}}55">
      ${{p.risk_emoji}} ${{p.risk_label}}
    </div>
    ${{p.hidden_risk ? `<div style="background:#e74c3c; color:white; padding:4px; font-size:0.75rem; border-radius:4px; margin-bottom:8px; font-weight:bold; text-align:center;">🔥 Tren Memburuk (Prioritas!)</div>` : ''}}
    <div class="popup-row"><span class="popup-key">Danger Score</span><span class="popup-val">${{ds}} / 100</span></div>
    <div class="popup-row"><span class="popup-key">Jml. Kejahatan</span><span class="popup-val">${{p.jumlah_kejahatan !== null ? Math.round(p.jumlah_kejahatan) : '—'}}</span></div>
    <div class="popup-row"><span class="popup-key">Risiko / 100k</span><span class="popup-val">${{p.risiko_100k !== null ? p.risiko_100k.toFixed(1) : '—'}}</span></div>
    <div class="popup-row"><span class="popup-key">% Penyelesaian</span><span class="popup-val">${{p.persen_selesai !== null ? p.persen_selesai.toFixed(2) + ' %' : '—'}}</span></div>
    <div class="popup-row"><span class="popup-key">Selang Waktu</span><span class="popup-val">${{p.selang_waktu !== null ? Math.round(p.selang_waktu) + ' jam' : '—'}}</span></div>
  `;
}}

// ─── SIDE INFO-PANEL ─────────────────────────────────────────────────────────
function updateInfoPanel(p) {{
  const ds  = p.danger_score !== null ? p.danger_score.toFixed(1) : '—';
  const stg = STRATEGIES[p.risk_label];
  const actionsHtml = stg ? stg.actions.map((a, i) => `
    <li style="border-left-color:${{stg.color}}">
      <span class="s-num" style="background:${{stg.color}}22; color:${{stg.color}}">${{i+1}}</span>
      <span>${{a}}</span>
    </li>`).join('') : '';

  document.getElementById('info-panel').innerHTML = `
    <div class="info-panel">
      <div class="ip-name" style="color:${{p.risk_color}}">${{p.risk_emoji}} ${{p.name}}</div>
      <div class="ip-row"><span class="ip-key">Profil Risiko</span><span class="ip-value">${{p.risk_label}}</span></div>
      <div class="ip-row"><span class="ip-key">Danger Score</span><span class="ip-value">${{ds}} / 100</span></div>
      <div class="ip-row"><span class="ip-key">Jml. Kejahatan</span><span class="ip-value">${{p.jumlah_kejahatan !== null ? Math.round(p.jumlah_kejahatan) : '—'}}</span></div>
      <div class="ip-row"><span class="ip-key">Risiko / 100k pend.</span><span class="ip-value">${{p.risiko_100k !== null ? p.risiko_100k.toFixed(1) : '—'}}</span></div>
      <div class="ip-row"><span class="ip-key">Tren Risiko 2020-2024</span><span class="ip-value" style="color:${{p.slope_risiko > 0 ? '#e74c3c' : '#2ecc71'}}">${{p.slope_risiko !== null ? (p.slope_risiko > 0 ? '↗ +' : '↘ ') + p.slope_risiko.toFixed(2) : '—'}}</span></div>
      <div class="ip-row"><span class="ip-key">% Penyelesaian</span><span class="ip-value">${{p.persen_selesai !== null ? p.persen_selesai.toFixed(2) + ' %' : '—'}}</span></div>
      <div class="ip-row"><span class="ip-key">Selang Waktu</span><span class="ip-value">${{p.selang_waktu !== null ? Math.round(p.selang_waktu) + ' jam' : '—'}}</span></div>
    </div>
    ${{stg ? `
    <div class="strategy-panel">
      <div class="strategy-header">
        <span class="strategy-icon">${{stg.icon}}</span>
        <span class="strategy-title">Saran Strategis</span>
      </div>
      <div class="strategy-subtitle">${{stg.subtitle}}</div>
      <ul class="strategy-list">${{actionsHtml}}</ul>
    </div>` : ''}}
  `;
}}

// ─── GeoJSON LAYER ───────────────────────────────────────────────────────────
const geoLayer = L.geoJSON(GEO, {{
  style: styleDefault,
  onEachFeature: function(feature, layer) {{
    const p = feature.properties;

    layer.on({{
      mouseover: function(e) {{
        if (activeLayer !== layer) {{
          layer.setStyle(styleHighlight(feature));
        }}
        layer.bringToFront();
      }},
      mouseout: function(e) {{
        if (activeLayer !== layer) {{
          geoLayer.resetStyle(layer);
          geoLayer.eachLayer(l => l.setStyle(styleDefault(l.feature)));
        }}
      }},
      click: function(e) {{
        if (activeLayer && activeLayer !== layer) {{
          geoLayer.resetStyle(activeLayer);
          geoLayer.eachLayer(l => l.setStyle(styleDefault(l.feature)));
        }}
        activeLayer = layer;
        layer.setStyle(styleHighlight(feature));
        layer.bringToFront();
        // panTo center tanpa zoom (fitBounds bisa menyebabkan black screen di region kecil)
        map.panTo(layer.getBounds().getCenter(), {{animate: true, duration: 0.4}});
        updateInfoPanel(p);
        // paksa Leaflet re-render ukuran peta setelah panel update
        setTimeout(() => map.invalidateSize(), 50);
        // scroll sidebar into view
        document.getElementById('info-panel').scrollIntoView({{behavior:'smooth', block:'nearest'}});
      }},
    }});

    layer.bindPopup(makePopup(p), {{
      maxWidth: 280,
      className: 'dark-popup',
    }});
  }},
}}).addTo(map);

// fit to all
map.fitBounds(geoLayer.getBounds(), {{padding: [20, 20]}});

// refresh style to apply highlightedRisk
function refreshStyles() {{
  geoLayer.eachLayer(l => l.setStyle(styleDefault(l.feature)));
}}

// ─── LEGEND SIDEBAR ──────────────────────────────────────────────────────────
const legendContainer = document.getElementById('legend-cards');

LEGEND.forEach((item, idx) => {{
  const card = document.createElement('div');
  card.className = 'legend-card';
  card.style.marginBottom = '10px';

  const membersHtml = item.members
    .map(m => `<span>${{m}}</span>`)
    .join('&nbsp;·&nbsp; ');

  const stg = STRATEGIES[item.label];
  card.innerHTML = `
    <div class="legend-card-header">
      <div class="risk-dot" style="background:${{item.color}}; color:${{item.color}}"></div>
      <span class="risk-name">${{item.emoji}} ${{item.label}}</span>
      <span class="risk-badge">${{item.n}} daerah</span>
    </div>
    <div class="stat-grid">
      ${{Object.entries(item.stats).map(([k,v]) => `
        <div class="stat-item">
          <div class="stat-label">${{k}}</div>
          <div class="stat-val">${{v}}</div>
        </div>`).join('')}}
    </div>
    <div class="danger-bar-wrap">
      <div class="danger-bar-label">
        <span>Danger Score</span>
        <span>${{item.danger.toFixed(1)}} / 100</span>
      </div>
      <div class="danger-bar-bg">
        <div class="danger-bar-fill" style="width:${{item.danger}}%; background:${{item.color}}"></div>
      </div>
    </div>
    ${{stg ? `
    <div class="card-strategy" style="border-left-color:${{item.color}}">
      <strong style="color:${{item.color}}">${{stg.icon}} ${{stg.title}}</strong>
      ${{stg.actions.slice(0, 2).map(a => '• ' + a).join('<br>')}}
      <span style="color:var(--accent); font-size:0.67rem;"> +${{stg.actions.length - 2}} lainnya...</span>
    </div>` : ''}}
    <div class="members-toggle" data-idx="${{idx}}">▸ Tampilkan daerah</div>
    <div class="members-list" id="mem-${{idx}}">${{membersHtml}}</div>
  `;

  // click on card → highlight map
  card.addEventListener('click', function(e) {{
    if (e.target.classList.contains('members-toggle')) return;
    const isActive = card.classList.contains('active');
    document.querySelectorAll('.legend-card').forEach(c => c.classList.remove('active'));
    if (isActive) {{
      highlightedRisk = null;
      card.classList.remove('active');
    }} else {{
      highlightedRisk = item.label;
      card.classList.add('active');
    }}
    activeLayer = null;
    refreshStyles();
  }});

  // toggle members
  card.querySelector('.members-toggle').addEventListener('click', function(e) {{
    e.stopPropagation();
    const list   = document.getElementById(`mem-${{idx}}`);
    const toggle = this;
    if (list.classList.contains('open')) {{
      list.classList.remove('open');
      toggle.textContent = '▸ Tampilkan daerah';
    }} else {{
      list.classList.add('open');
      toggle.textContent = '▾ Sembunyikan';
    }}
  }});

  legendContainer.appendChild(card);
}});
</script>
</body>
</html>
"""

with open(OUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print(f"\n✅ Peta berhasil dibuat: {OUT_HTML}")
print(f"   Buka file tersebut di browser untuk melihat visualisasi interaktif.")
