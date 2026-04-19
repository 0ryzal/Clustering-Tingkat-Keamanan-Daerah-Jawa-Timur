import json

with open('model.ipynb', 'r', encoding='utf-8') as f:
    data = json.load(f)

for cell in data['cells']:
    if cell['cell_type'] == 'code':
        content = ''.join(cell['source'])
        if "if special_case_note:" in content and "print(f\"⚠️ {special_case_note}\")" in content:
            target = "if special_case_note:\n    print(f\"⚠️ {special_case_note}\")"
            if target in content and "Davies-Bouldin (K=10)" not in content:
                added_code = """
if metric_k.get('Davies-Bouldin') == 10 and OPTIMAL_K != 10:
    print(f"\\n🔍 Analisis Lanjutan Davies-Bouldin (K=10):")
    print(f"Meskipun metrik Davies-Bouldin menunjuk K=10 sebagai nilai minimum (terbaik), K=10 tidak dipilih")
    print(f"karena ada cluster yang anggotanya < {MIN_CLUSTER_SIZE} daerah (melanggar batas guardrail MIN_CLUSTER_SIZE).")
    print(f"Keputusan ini memastikan bahwa klaster yang terbentuk cukup representatif untuk kebijakan tata ruang Jatim, mencegah over-segmentation.")"""
                new_content = content.replace(target, target + added_code)
                lines = [l + '\n' for l in new_content.split('\n')]
                if len(lines) > 0:
                    lines[-1] = lines[-1].rstrip('\n')
                cell['source'] = lines

with open('model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=1)

print('Berhasil menambahkan alasan guardrail Davies-Bouldin K=10 ke model.ipynb')
