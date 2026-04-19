"""
Script untuk membuat visualisasi peta STATIS tingkat kriminalitas Jawa Timur.
Menggunakan geopandas + matplotlib, bergaya seperti gambar referensi.
Output: peta_kriminalitas_statis.png

CATATAN: Assignment cluster menggunakan hasil langsung dari model.ipynb
         agar 100% konsisten (tidak re-run K-Means yang bisa berubah).
"""

import warnings
warnings.filterwarnings("ignore")

try:
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    print("✅ geopandas & matplotlib tersedia")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("  Jalankan: pip install geopandas matplotlib")
    exit(1)

from collections import Counter

# ─────────────────────────────────────────────────────────────────────────────
# 1. KONFIGURASI
# ─────────────────────────────────────────────────────────────────────────────
GEOJSON = "Kabupaten-Kota (Provinsi Jawa Timur).geojson"
OUT_PNG = "peta_kriminalitas_statis.png"

# ─────────────────────────────────────────────────────────────────────────────
# 2. HARDCODED CLUSTER ASSIGNMENT DARI MODEL.IPYNB
#    Sumber: output K-Means + Danger Score di model.ipynb (K=4)
#    Format: "Nama Kab/Kota" : ("Risk Label", "#HexColor", "Emoji")
# ─────────────────────────────────────────────────────────────────────────────
CLUSTER_MAP = {
    # 🔴 High Risk – Low Resolution (Klaster 2, Danger Score ~47)
    "Banyuwangi"    : ("High Risk", "#e74c3c", "🔴"),
    "Bojonegoro"    : ("High Risk", "#e74c3c", "🔴"),
    "Gresik"        : ("High Risk", "#e74c3c", "🔴"),
    "Jember"        : ("High Risk", "#e74c3c", "🔴"),
    "Jombang"       : ("High Risk", "#e74c3c", "🔴"),
    "Kediri"        : ("High Risk", "#e74c3c", "🔴"),
    "Kota Surabaya" : ("High Risk", "#e74c3c", "🔴"),
    "Lamongan"      : ("High Risk", "#e74c3c", "🔴"),
    "Malang"        : ("High Risk", "#e74c3c", "🔴"),
    "Pasuruan"      : ("High Risk", "#e74c3c", "🔴"),
    "Sidoarjo"      : ("High Risk", "#e74c3c", "🔴"),
    "Tulungagung"   : ("High Risk", "#e74c3c", "🔴"),

    # 🟠 Moderate-High Risk – Needs Attention (Klaster 4, Danger Score ~38)
    "Kota Kediri"      : ("Moderate-High Risk", "#e67e22", "🟠"),
    "Kota Mojokerto"   : ("Moderate-High Risk", "#e67e22", "🟠"),
    "Kota Pasuruan"    : ("Moderate-High Risk", "#e67e22", "🟠"),
    "Kota Probolinggo" : ("Moderate-High Risk", "#e67e22", "🟠"),
    "Magetan"          : ("Moderate-High Risk", "#e67e22", "🟠"),
    "Ngawi"            : ("Moderate-High Risk", "#e67e22", "🟠"),

    # 🟡 Moderate-Low Risk – Watch Zone (Klaster 3, Danger Score ~35)
    "Bondowoso"  : ("Moderate-Low Risk", "#f1c40f", "🟡"),
    "Kota Malang": ("Moderate-Low Risk", "#f1c40f", "🟡"),
    "Lumajang"   : ("Moderate-Low Risk", "#f1c40f", "🟡"),
    "Mojokerto"  : ("Moderate-Low Risk", "#f1c40f", "🟡"),
    "Pamekasan"  : ("Moderate-Low Risk", "#f1c40f", "🟡"),
    "Situbondo"  : ("Moderate-Low Risk", "#f1c40f", "🟡"),
    "Sumenep"    : ("Moderate-Low Risk", "#f1c40f", "🟡"),
    "Tuban"      : ("Moderate-Low Risk", "#f1c40f", "🟡"),

    # 🟢 Safe Zone – Low Priority (Klaster 1, Danger Score ~23)
    "Bangkalan"   : ("Safe Zone", "#2ecc71", "🟢"),
    "Blitar"      : ("Safe Zone", "#2ecc71", "🟢"),
    "Kota Batu"   : ("Safe Zone", "#2ecc71", "🟢"),
    "Kota Blitar" : ("Safe Zone", "#2ecc71", "🟢"),
    "Kota Madiun" : ("Safe Zone", "#2ecc71", "🟢"),
    "Madiun"      : ("Safe Zone", "#2ecc71", "🟢"),
    "Nganjuk"     : ("Safe Zone", "#2ecc71", "🟢"),
    "Pacitan"     : ("Safe Zone", "#2ecc71", "🟢"),
    "Ponorogo"    : ("Safe Zone", "#2ecc71", "🟢"),
    "Probolinggo" : ("Safe Zone", "#2ecc71", "🟢"),
    "Sampang"     : ("Safe Zone", "#2ecc71", "🟢"),
    "Trenggalek"  : ("Safe Zone", "#2ecc71", "🟢"),
}

# Cetak ringkasan untuk verifikasi
label_counts = Counter(v[0] for v in CLUSTER_MAP.values())
print("✅ Distribusi kluster (dari model.ipynb):")
for label, count in sorted(label_counts.items()):
    emoji   = next(v[2] for v in CLUSTER_MAP.values() if v[0] == label)
    members = sorted(k for k, v in CLUSTER_MAP.items() if v[0] == label)
    print(f"   {emoji} {label} ({count} daerah): {', '.join(members)}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. LOAD GEOJSON & PETAKAN KE CLUSTER
# ─────────────────────────────────────────────────────────────────────────────
# Beberapa nama berbeda antara dataset & GeoJSON
NAME_MAP = {
    "Kota Batu"    : "Batu",       # GeoJSON pakai "Batu"
    "Kota Surabaya": "Surabaya",   # GeoJSON pakai "Surabaya"
}
GEO_TO_DATA = {v: k for k, v in NAME_MAP.items()}

gdf = gpd.read_file(GEOJSON)
print(f"\n✅ GeoJSON dimuat: {len(gdf)} fitur, CRS={gdf.crs}")

def get_risk_color(geo_name):
    data_name = GEO_TO_DATA.get(geo_name, geo_name)
    info = CLUSTER_MAP.get(data_name)
    return info[1] if info else "#AAAAAA"

def get_risk_label(geo_name):
    data_name = GEO_TO_DATA.get(geo_name, geo_name)
    info = CLUSTER_MAP.get(data_name)
    return info[0] if info else "No Data"

gdf["Risk_Color"] = gdf["NAME_2"].apply(get_risk_color)
gdf["Risk_Label"] = gdf["NAME_2"].apply(get_risk_label)

# Cek nama yang tidak cocok
no_data = gdf[gdf["Risk_Label"] == "No Data"]["NAME_2"].tolist()
if no_data:
    print(f"⚠️  Nama tidak cocok di GeoJSON: {no_data}")
else:
    print("✅ Semua nama wilayah berhasil dipetakan ke kluster")

# ─────────────────────────────────────────────────────────────────────────────
# 4. PLOT PETA STATIS
# ─────────────────────────────────────────────────────────────────────────────
HIGH_RISK_GEO = [geo for geo in gdf["NAME_2"] if get_risk_label(geo) == "High Risk"]

fig, ax = plt.subplots(1, 1, figsize=(18, 9))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Gambar polygon
gdf.plot(ax=ax, color=gdf["Risk_Color"], edgecolor="#FFFFFF", linewidth=0.8)

# Label kuning untuk wilayah High Risk
for idx, row in gdf.iterrows():
    name = row["NAME_2"]
    if name not in HIGH_RISK_GEO:
        continue
    try:
        centroid = row["geometry"].centroid
        cx, cy = centroid.x, centroid.y
    except Exception:
        continue

    display_name  = GEO_TO_DATA.get(name, name)
    display_short = display_name.replace("Kabupaten ", "KAB. ").upper()

    ax.annotate(
        display_short,
        xy=(cx, cy),
        xytext=(cx, cy + 0.15),
        ha="center", va="bottom",
        fontsize=7.5, fontweight="bold", color="#1a1a1a",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="#FFFF66", edgecolor="#999900",
            linewidth=0.8, alpha=0.92,
        ),
        arrowprops=dict(arrowstyle="-", color="#555555", lw=0.6),
        zorder=5,
    )

# Judul
ax.set_title(
    "Peta Klaster Tingkat Kriminalitas\nKabupaten/Kota Provinsi Jawa Timur (2020–2024)",
    fontsize=16, fontweight="bold", pad=15, color="#1a1a1a"
)

# Legenda
legend_data = [
    ("High Risk",          "#e74c3c"),
    ("Moderate-High Risk", "#e67e22"),
    ("Moderate-Low Risk",  "#f1c40f"),
    ("Safe Zone",          "#2ecc71"),
]
patches = [
    mpatches.Patch(color=color, label=label, linewidth=0.5, edgecolor="white")
    for label, color in legend_data
]
leg = ax.legend(
    handles=patches,
    title="Profil Risiko Klaster",
    title_fontsize=9, fontsize=8.5,
    loc="lower left",
    framealpha=0.9, frameon=True,
    edgecolor="#CCCCCC", facecolor="white",
)
leg.get_title().set_fontweight("bold")

ax.set_axis_off()
plt.tight_layout(pad=1.0)
plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight", facecolor="white")
print(f"\n✅ Peta statis disimpan ke: {OUT_PNG}")
plt.show()
