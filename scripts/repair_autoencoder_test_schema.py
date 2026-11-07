"""Repair the v2 AE/VAE test CSV's omitted probe_n_iter header, preserving rows."""
import csv
from pathlib import Path
root=Path(__file__).resolve().parents[1]; path=root/'results/raw/autoencoder_test_results.csv'
with path.open(encoding='utf-8-sig',newline='') as f: data=list(csv.reader(f))
header=data[0]
if 'probe_n_iter' not in header:
    header.insert(header.index('probe_evaluation_seconds'),'probe_n_iter')
fixed=[header]
for row in data[1:]:
    if len(row)==len(header): fixed.append(row)
    elif len(row)==len(header)-1:
        row.insert(header.index('probe_evaluation_seconds'),''); fixed.append(row)
    else: raise ValueError(f'Unexpected field count {len(row)} for row {row[:2]}')
with path.open('w',encoding='utf-8-sig',newline='') as f: csv.writer(f).writerows(fixed)
print(f'Repaired header; preserved {len(fixed)-1} data rows.')
