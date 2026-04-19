import json

with open('model.ipynb', 'r', encoding='utf-8') as f:
    data = json.load(f)

for cell in data['cells']:
    if cell['cell_type'] == 'code':
        content = ''.join(cell['source'])
        if 'kmeans_silh = f"{sil_scores[OPTIMAL_K]:.4f}"' in content:
            new_content = content
            new_content = new_content.replace('kmeans_silh = f"{sil_scores[OPTIMAL_K]:.4f}"', 'kmeans_silh = f"{silh_km:.4f}"')
            new_content = new_content.replace('kmeans_silh = "[Nilai terhitung di cell sebelumnya]"', 'kmeans_silh = "[jalankan Cell 28]"')
            new_content = new_content.replace('kmeans_silh = "[Nilai terhitung...]"', 'kmeans_silh = "[jalankan Cell 28]"') # jic
            
            new_content = new_content.replace(
                'hc_silh = f"{hc_score:.4f}" if \'hc_score\' in globals() else "[Nilai terhitung di cell sebelumnya]"',
                'hc_silh = f"{silh_hc:.4f}" \\\n        if \'silh_hc\' in globals() \\\n        else "[jalankan Cell 34]"'
            )
            new_content = new_content.replace(
                'hc_silh = f"{hc_score:.4f}" \\\n        if \'hc_score\' in globals() \\\n        else "[Nilai terhitung...]"',
                'hc_silh = f"{silh_hc:.4f}" \\\n        if \'silh_hc\' in globals() \\\n        else "[jalankan Cell 34]"'
            )

            new_content = new_content.replace('hc_silh = "[Nilai terhitung di cell sebelumnya]"', 'hc_silh = "[jalankan Cell 34]"')
            new_content = new_content.replace('hc_silh = "[Nilai terhitung...]"', 'hc_silh = "[jalankan Cell 34]"')

            new_content = new_content.replace(
                'ari_val = f"{optimal_ari:.4f}" if \'optimal_ari\' in globals() else "[Nilai terhitung di cell sebelumnya]"',
                'ari_val = f"{ari:.4f}" \\\n        if \'ari\' in globals() \\\n        else "[jalankan Cell 34]"'
            )
            new_content = new_content.replace(
                'ari_val = f"{optimal_ari:.4f}" \\\n        if \'optimal_ari\' in globals() \\\n        else "[Nilai terhitung...]"',
                'ari_val = f"{ari:.4f}" \\\n        if \'ari\' in globals() \\\n        else "[jalankan Cell 34]"'
            )

            new_content = new_content.replace('ari_val = "[Nilai terhitung di cell sebelumnya]"', 'ari_val = "[jalankan Cell 34]"')
            new_content = new_content.replace('ari_val = "[Nilai terhitung...]"', 'ari_val = "[jalankan Cell 34]"')

            lines = [l + '\n' for l in new_content.split('\n')]
            if len(lines) > 0:
                lines[-1] = lines[-1].rstrip('\n')
                
            cell['source'] = lines

with open('model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=1)

print('Berhasil memperbarui variabel silh_km, silh_hc, dan ari di model.ipynb')
