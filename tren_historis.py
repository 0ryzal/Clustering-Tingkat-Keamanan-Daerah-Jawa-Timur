"""
Visualisasi Tren Historis Tingkat Kriminalitas Jawa Timur 2020-2024
Bergaya seperti gambar referensi: garis tren + area shading + annotasi nilai + R²
Output: tren_historis_kriminalitas.png
"""

import glob
import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. KONFIGURASI
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = "tingkat kriminalitas"
OUT_PNG  = "tren_historis_kriminalitas.png"

COL_NAMES    = ["Kabupaten_Kota", "Jumlah_Kejahatan", "Risiko_100k",
                "Persen_Penyelesaian", "Selang_Waktu"]
FEATURE_COLS = COL_NAMES[1:]

# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD DATA PER TAHUN
# ─────────────────────────────────────────────────────────────────────────────
def load_year_data(filepath):
    year = int(os.path.basename(filepath).split()[-1].replace(".xlsx", ""))
    df   = pd.read_excel(filepath, header=0)
    df.columns = COL_NAMES

    # Filter baris Jawa Timur (provinsi)
    mask_jatim = df["Kabupaten_Kota"].astype(str).str.lower().str.contains("jawa timur", na=True)
    df = df[~mask_jatim].copy()

    # Konversi ke numerik
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Hapus baris metadata
    df = df[df["Kabupaten_Kota"].notna()].copy()
    df = df[~df["Kabupaten_Kota"].astype(str).str.match(
        r"^[<\d]|catatan|sumber|source|note", flags=0, na=True
    )].copy()

    valid_mask = df[FEATURE_COLS].notna().sum(axis=1) >= 3
    df = df[valid_mask].copy()
    df["Tahun"] = year
    return df

FILES  = sorted(glob.glob(os.path.join(DATA_DIR, "*.xlsx")))
dfs    = [load_year_data(f) for f in FILES]
df_raw = pd.concat(dfs, ignore_index=True)
print(f"✅ Data dimuat: {len(FILES)} tahun, {df_raw['Kabupaten_Kota'].nunique()} Kab/Kota")

# ─────────────────────────────────────────────────────────────────────────────
# 3. AGREGASI: TOTAL / RATA-RATA JAWA TIMUR PER TAHUN
# ─────────────────────────────────────────────────────────────────────────────
# Jumlah Kejahatan → total (SUM) semua kab/kota
# Risiko/100k, % Penyelesaian, Selang Waktu → rata-rata (MEAN) semua kab/kota
agg_dict = {
    "Jumlah_Kejahatan"   : "sum",
    "Risiko_100k"        : "mean",
    "Persen_Penyelesaian": "mean",
    "Selang_Waktu"       : "mean",
}
df_trend = df_raw.groupby("Tahun").agg(agg_dict).reset_index()
df_trend = df_trend.sort_values("Tahun").reset_index(drop=True)

print("\n📊 Data Tren per Tahun:")
print(df_trend.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 4. HITUNG REGRESI LINEAR & R² UNTUK SETIAP VARIABEL
# ─────────────────────────────────────────────────────────────────────────────
years = df_trend["Tahun"].values
x = np.arange(len(years))   # 0,1,2,3,4

def compute_regression(y_vals):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y_vals)
    r_sq = r_value ** 2
    y_pred = slope * x + intercept
    delta_per_year = slope
    return slope, intercept, r_sq, y_pred, delta_per_year

# ─────────────────────────────────────────────────────────────────────────────
# 5. KONFIGURASI VISUAL TIAP PANEL
# ─────────────────────────────────────────────────────────────────────────────
panels = [
    {
        "col"      : "Jumlah_Kejahatan",
        "title"    : "Total Jumlah Kejahatan\nProvinsi Jawa Timur 2020–2024",
        "ylabel"   : "Jumlah Kejahatan (kasus)",
        "color"    : "#e74c3c",     # merah
        "fill"     : "#f9c6c6",
        "fmt"      : lambda v: f"{v:,.0f}",
        "delta_lbl": "kasus/tahun",
    },
    {
        "col"      : "Risiko_100k",
        "title"    : "Rata-rata Risiko per 100k Penduduk\nProvinsi Jawa Timur 2020–2024",
        "ylabel"   : "Risiko / 100k Penduduk",
        "color"    : "#e67e22",     # oranye
        "fill"     : "#fde8cc",
        "fmt"      : lambda v: f"{v:.1f}",
        "delta_lbl": "pt/tahun",
    },
    {
        "col"      : "Persen_Penyelesaian",
        "title"    : "Rata-rata Persentase Penyelesaian Kasus\nProvinsi Jawa Timur 2020–2024",
        "ylabel"   : "% Penyelesaian Kasus",
        "color"    : "#2ecc71",     # hijau
        "fill"     : "#c9f7dd",
        "fmt"      : lambda v: f"{v:.1f}%",
        "delta_lbl": "pp/tahun",
    },
    {
        "col"      : "Selang_Waktu",
        "title"    : "Rata-rata Selang Waktu Kejadian\nProvinsi Jawa Timur 2020–2024",
        "ylabel"   : "Selang Waktu (jam)",
        "color"    : "#3498db",     # biru
        "fill"     : "#c8e6f8",
        "fmt"      : lambda v: f"{v:.0f}j",
        "delta_lbl": "jam/tahun",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# 6. PLOT
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(22, 6))
fig.patch.set_facecolor("white")
fig.suptitle(
    "Tren Historis Indikator Kriminalitas Provinsi Jawa Timur (2020–2024)",
    fontsize=15, fontweight="bold", y=1.02, color="#1a1a1a"
)

for ax, panel in zip(axes, panels):
    col     = panel["col"]
    color   = panel["color"]
    fill    = panel["fill"]
    fmt     = panel["fmt"]

    y_vals  = df_trend[col].values
    slope, intercept, r_sq, y_pred, delta = compute_regression(y_vals)

    # ── Area shading ──────────────────────────────────────────────────────
    ax.fill_between(years, y_vals, alpha=0.25, color=fill, zorder=1)
    ax.fill_between(years, y_pred, alpha=0.15, color=color, zorder=1)

    # ── Garis tren (putus-putus) ──────────────────────────────────────────
    ax.plot(years, y_pred, linestyle="--", color=color, linewidth=1.8,
            alpha=0.85, label=f"Trend linear (R²={r_sq:.3f})", zorder=3)

    # ── Garis aktual ─────────────────────────────────────────────────────
    ax.plot(years, y_vals, marker="o", markersize=8, color=color,
            linewidth=2.2, markerfacecolor="white", markeredgewidth=2,
            label=col.replace("_", " "), zorder=4)

    # ── Anotasi nilai di setiap titik ────────────────────────────────────
    for yr, val in zip(years, y_vals):
        label_str = fmt(val)
        ax.annotate(
            label_str,
            xy=(yr, val),
            xytext=(0, 11),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=9, fontweight="bold",
            color=color,
        )

    # ── Legenda R² di pojok kiri atas ─────────────────────────────────────
    ax.legend(
        loc="upper left", fontsize=8,
        framealpha=0.85, edgecolor="#cccccc",
        facecolor="white"
    )

    # ── Anotasi delta per tahun di pojok kanan bawah ──────────────────────
    delta_sign = "+" if delta >= 0 else ""
    delta_txt  = f"Δ/tahun = {delta_sign}{fmt(delta).replace('%','').replace('j','')} {panel['delta_lbl']}\nR² = {r_sq:.3f}"
    ax.annotate(
        delta_txt,
        xy=(0.97, 0.05),
        xycoords="axes fraction",
        ha="right", va="bottom",
        fontsize=8,
        color="#1a1a1a",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#f0f0f0",
            edgecolor="#aaaaaa",
            alpha=0.9,
        ),
    )

    # ── Styling sumbu ─────────────────────────────────────────────────────
    ax.set_title(panel["title"], fontsize=11, fontweight="bold",
                 color="#1a1a1a", pad=10)
    ax.set_xlabel("Tahun", fontsize=9, color="#555555")
    ax.set_ylabel(panel["ylabel"], fontsize=9, color="#555555")
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years], fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_facecolor("white")
    ax.grid(axis="y", linestyle="--", alpha=0.4, color="#cccccc")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")

    # ── Padding Y agar anotasi tidak terpotong ────────────────────────────
    y_min = min(y_vals.min(), y_pred.min())
    y_max = max(y_vals.max(), y_pred.max())
    margin = (y_max - y_min) * 0.20
    ax.set_ylim(y_min - margin, y_max + margin * 1.5)

# Footer sumber
fig.text(
    0.5, -0.03,
    "Sumber: BPS Jawa Timur — Data Statistik Kriminalitas 2020–2024",
    ha="center", fontsize=9, color="#777777", style="italic"
)

plt.tight_layout(pad=2.0)
plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight", facecolor="white")
print(f"\n✅ Grafik tren historis disimpan ke: {OUT_PNG}")
plt.show()
