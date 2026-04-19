import codecs

with codecs.open("model.ipynb", "r", "utf-8") as f:
    text = f.read()

# Replace TOC
text = text.replace(
    '    "8. [Profil Cluster & Pelabelan Risiko](#8-profiling)\\n",\n    "9. [Rekomendasi Strategis](#9-rekomendasi)"',
    '    "8. [Profil Cluster & Pelabelan Risiko](#8-profiling)\\n",\n    "9. [Rekomendasi Strategis](#9-rekomendasi)\\n",\n    "10. [Tren Temporal](#10-tren)\\n",\n    "11. [Peta Spasial](#11-peta)\\n",\n    "12. [Limitasi](#12-limitasi)"'
)

# Replace section 10
text = text.replace(
    '    "## 10. Analisis Lapisan Tambahan: Tren Temporal Tersembunyi (2020-2024)\\n",\n    "\\n",',
    '    "## 10. Analisis Lapisan Tambahan: Tren Temporal Tersembunyi (2020-2024) <a id=\'10-tren\'></a>\\n",\n    "\\n",'
)

# Replace section 11
text = text.replace(
    '    "## 11. Visualisasi Spasial Peta Interaktif 🗺️\\n",\n    "        \\n",',
    '    "## 11. Visualisasi Spasial Peta Interaktif 🗺️ <a id=\'11-peta\'></a>\\n",\n    "        \\n",'
)

# Replace section 12
text = text.replace(
    '    "## 12. Limitasi & Catatan Metodologi\\n",\n    "\\n",',
    '    "## 12. Limitasi & Catatan Metodologi <a id=\'12-limitasi\'></a>\\n",\n    "\\n",'
)

with codecs.open("model.ipynb", "w", "utf-8") as f:
    f.write(text)

print("done")
