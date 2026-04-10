"""
Script untuk membuat Heatmap Korelasi Antar Variabel
Menggunakan data dari model.ipynb dengan styling mirip gambar referensi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# ── Setup ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi'      : 130,
    'font.family'     : 'DejaVu Sans',
    'axes.titlesize'  : 13,
    'axes.labelsize'  : 11,
    'axes.spines.top' : False,
    'axes.spines.right': False,
})

# ── Konfigurasi ─────────────────────────────────────────────────────────────
DATA_DIR  = "tingkat kriminalitas"
FILES     = sorted(glob.glob(os.path.join(DATA_DIR, "*.xlsx")))

COL_NAMES    = ['Kabupaten_Kota', 'Jumlah_Kejahatan', 'Risiko_100k',
                'Persen_Penyelesaian', 'Selang_Waktu']
FEATURE_COLS = COL_NAMES[1:]

FEATURE_LABELS = {
    'Jumlah_Kejahatan'   : 'Jml Kejahatan',
    'Risiko_100k'        : 'Risiko/100k',
    'Persen_Penyelesaian': '% Penyelesaian',
    'Selang_Waktu'       : 'Selang Waktu',
}


def load_year_data(filepath: str) -> pd.DataFrame:
    """Load data dari satu file Excel dan clean"""
    year = int(os.path.basename(filepath).split()[-1].replace('.xlsx', ''))
    df   = pd.read_excel(filepath, header=0)
    df.columns = COL_NAMES

    # Filter baris 'Jawa Timur'
    mask_jatim = df['Kabupaten_Kota'].astype(str).str.lower().str.contains('jawa timur', na=True)
    df = df[~mask_jatim].copy()

    # Konversi ke numerik
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Hapus baris metadata
    df = df[df['Kabupaten_Kota'].notna()].copy()
    df = df[~df['Kabupaten_Kota'].astype(str).str.match(
        r'^[<\d]|catatan|sumber|source|note', flags=0, na=True
    )].copy()

    # Hanya simpan baris dengan ≥ 3 kolom numerik valid
    valid_mask = df[FEATURE_COLS].notna().sum(axis=1) >= 3
    df = df[valid_mask].copy()

    df['Tahun'] = year
    return df


# ── Load & Aggregasi Data ────────────────────────────────────────────────────
dfs    = [load_year_data(f) for f in FILES]
df_raw = pd.concat(dfs, ignore_index=True)
df_raw[FEATURE_COLS] = df_raw[FEATURE_COLS].apply(pd.to_numeric, errors='coerce')

# Agregasi median per Kabupaten/Kota
df_profile = (df_raw
              .groupby('Kabupaten_Kota', sort=True)[FEATURE_COLS]
              .median()
              .reset_index())

print(f"✅ Data loaded: {len(df_profile)} Kabupaten/Kota")


# ── Hitung Korelasi ─────────────────────────────────────────────────────────
corr_matrix = df_profile[FEATURE_COLS].corr()

print("\n📊 Matriks Korelasi:")
print(corr_matrix.round(3).to_string())
print()


# ── Buat Heatmap dengan Styling Mirip Gambar Referensi ────────────────────────
fig, ax = plt.subplots(figsize=(14, 11))

# Heatmap dengan colorbar style
im = ax.imshow(
    corr_matrix.values,
    cmap='coolwarm',
    vmin=-1,
    vmax=1,
    aspect='auto'
)

# Set ticks dan labels
ax.set_xticks(np.arange(len(FEATURE_COLS)))
ax.set_yticks(np.arange(len(FEATURE_COLS)))
ax.set_xticklabels([FEATURE_LABELS[col] for col in FEATURE_COLS], fontsize=11)
ax.set_yticklabels([FEATURE_LABELS[col] for col in FEATURE_COLS], fontsize=11)

# Rotate x labels
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Add grid dan borders
for i in range(len(FEATURE_COLS)):
    for j in range(len(FEATURE_COLS)):
        ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                   fill=False, edgecolor='white', lw=1.5))

# Add correlation values di setiap cell
for i in range(len(FEATURE_COLS)):
    for j in range(len(FEATURE_COLS)):
        value = corr_matrix.iloc[i, j]
        text_color = 'white' if abs(value) > 0.6 else 'black'
        ax.text(j, i, f'{value:.2f}',
               ha="center", va="center", color=text_color,
               fontsize=12, fontweight='bold')

# Title
ax.set_title(
    'HEATMAP KORELASI ANTAR VARIABEL\nProvinsi Jawa Timur',
    fontsize=15,
    fontweight='bold',
    pad=20
)

# Colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Koefisien Korelasi', rotation=270, labelpad=20, fontsize=11)

# Figure text
fig.text(0.5, 0.01, 'Gambar. Heatmap Correlation Antar Variabel Keamanan',
        ha='center', fontsize=10, style='italic')

ax.set_aspect('equal', adjustable='box')
plt.tight_layout(rect=[0, 0.02, 1, 1])
plt.savefig('heatmap_korelasi_antar_variabel.png',
            bbox_inches='tight', dpi=130, facecolor='white')
print("✅ Heatmap tersimpan sebagai: heatmap_korelasi_antar_variabel.png")
plt.show()


# ── Analisis Interpretasi Korelasi ──────────────────────────────────────────
print("\n" + "="*70)
print("  INTERPRETASI KORELASI ANTAR VARIABEL")
print("="*70)

# Ekstrak korelasi dengan threshold
threshold = 0.5
for i in range(len(FEATURE_COLS)):
    for j in range(i+1, len(FEATURE_COLS)):
        corr_val = corr_matrix.iloc[i, j]
        col_i = FEATURE_LABELS[FEATURE_COLS[i]]
        col_j = FEATURE_LABELS[FEATURE_COLS[j]]

        if abs(corr_val) >= threshold:
            if corr_val > 0:
                strength = "kuat positif" if corr_val > 0.75 else "sedang positif"
            else:
                strength = "kuat negatif" if corr_val < -0.75 else "sedang negatif"
            print(f"\n  ✓ {col_i} ↔ {col_j}")
            print(f"    Korelasi: {corr_val:.3f} ({strength})")

print("\n" + "="*70)
