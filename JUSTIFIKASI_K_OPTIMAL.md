# Justifikasi K Optimal - Sensitivity K=4 vs K=9

## Ringkasan

Sensitivity analysis dilakukan untuk membandingkan konfigurasi **K=4** (lebih parsimonious)
dan **K=9** (lebih granular) pada metrik validasi internal dan interpretabilitas label risiko.

Aturan keputusan yang digunakan:

- Jika $|S_{K=9} - S_{K=4}| < 0.02$, maka dipilih **K=4** sebagai hasil utama (prinsip parsimony).
- Konfigurasi **K=9** ditetapkan sebagai lampiran sensitivity.
- Tambahan guardrail kebijakan: K yang dipakai sebagai basis kebijakan harus memiliki
    **minimum 4 anggota di setiap cluster**.
- Notebook menampilkan otomatis **K pertama** yang memenuhi syarat minimum anggota
    dan **K paling granular** yang masih memenuhi syarat tersebut.

Hasil guardrail pada rentang uji K=2-10:

- **K pertama** yang memenuhi minimum 4 anggota di semua cluster: **K=2**.
- **K paling granular** yang masih memenuhi minimum 4 anggota di semua cluster: **K=5**.

## Tabel Perbandingan K=4 vs K=9

| K | Silhouette | Calinski-Harabasz | Davies-Bouldin | Min Anggota Cluster | Cluster <4 Anggota | Rata-rata Cluster per Label Risiko | Interpretabilitas Label Risiko | Peran di Laporan |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 4 | 0.2829 | 13.8082 | 1.1184 | 6 | 0 | 1.00 | Tinggi | Hasil utama |
| 9 | 0.3027 | 12.5661 | 0.8834 | 2 | 3 | 2.25 | Rendah | Lampiran sensitivity |

## Keputusan Akhir

- Selisih silhouette: $|0.3027 - 0.2829| = 0.0198 < 0.02$.
- Karena pemisahan K=4 masih sejelas K=9 secara praktis, maka **K=4 dipilih sebagai hasil utama**.
- K=9 tetap dicantumkan sebagai **analisis sensitivitas** untuk menunjukkan granularitas tambahan,
  namun tidak dijadikan basis utama keputusan kebijakan karena muncul cluster sangat kecil
  (minimum 2 anggota, 3 cluster beranggota <4).
- Jika konfigurasi granular seperti K=9 tetap digunakan pada konteks tertentu,
  cluster beranggota <4 harus diperlakukan sebagai **kasus khusus** yang memerlukan
  verifikasi lapangan sebelum dijadikan acuan kebijakan operasional.
