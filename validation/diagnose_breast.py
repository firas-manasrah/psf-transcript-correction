"""
Diagnostic: check why marker genes returned 0 counts in the breast validation window.
Checks (1) whether the gene names actually exist in this panel, and
(2) whether the 500x500um window centred on the whole-slide transcript mean
actually falls on tissue.
"""
import pandas as pd
import json
from pathlib import Path

OUT = Path('/home/jovyan/scratch/xenium_breast/outs')

# 1. Check the gene panel itself
print('=== GENE PANEL CHECK ===')
with open(OUT / 'gene_panel.json') as f:
    panel = json.load(f)
panel_genes = set()
for target in panel.get('payload', {}).get('targets', []):
    name = target.get('type', {}).get('data', {}).get('name')
    if name:
        panel_genes.add(name)
print(f'Total panel targets: {len(panel_genes)}')

markers = ['EPCAM','ERBB2','ESR1','KRT14','ACTA2','PTPRC','PECAM1']
for g in markers:
    print(f'  {g}: {"IN PANEL" if g in panel_genes else "NOT IN PANEL"}')

# 2. Check actual feature_name values in transcripts.parquet (just a sample)
print('\n=== TRANSCRIPT FEATURE_NAME SAMPLE ===')
tx_sample = pd.read_parquet(str(OUT / 'transcripts.parquet'), columns=['feature_name'])
print(f'Total transcripts: {len(tx_sample):,}')
# BUG FIX: feature_name is stored as bytes (b'EPCAM'), not str ('EPCAM').
# Comparing bytes == str silently never matches, which is why every
# marker gene showed 0 transcripts despite being confirmed in-panel.
if len(tx_sample) > 0 and isinstance(tx_sample['feature_name'].iloc[0], bytes):
    print('(feature_name is bytes — decoding to str before comparison)')
    tx_sample['feature_name'] = tx_sample['feature_name'].str.decode('utf-8')
print('Unique feature_name count:', tx_sample['feature_name'].nunique())
print('Sample values:', tx_sample['feature_name'].drop_duplicates().head(20).tolist())

for g in markers:
    n = (tx_sample['feature_name'] == g).sum()
    print(f'  {g}: {n:,} total transcripts in whole dataset')

# 3. Check if the previous 500x500um window actually had tissue
print('\n=== WINDOW / TISSUE EXTENT CHECK ===')
tx = pd.read_parquet(str(OUT / 'transcripts.parquet'), columns=['x_location','y_location','qv'])
tx = tx[tx['qv'] >= 20]
print(f'Full-slide x range: {tx.x_location.min():.1f} to {tx.x_location.max():.1f} um')
print(f'Full-slide y range: {tx.y_location.min():.1f} to {tx.y_location.max():.1f} um')
print(f'Full-slide transcript mean (used as window centre last time): '
      f'x={tx.x_location.mean():.1f}, y={tx.y_location.mean():.1f}')

# Find the densest 500x500um tile instead — much more likely to hit real tissue
print('\nScanning for densest 500x500um tile...')
import numpy as np
xb = np.linspace(tx.x_location.min(), tx.x_location.max(), 20)
yb = np.linspace(tx.y_location.min(), tx.y_location.max(), 20)
H, xe, ye = np.histogram2d(tx.x_location, tx.y_location, bins=[xb, yb])
best = np.unravel_index(H.argmax(), H.shape)
best_x = (xe[best[0]] + xe[best[0]+1]) / 2
best_y = (ye[best[1]] + ye[best[1]+1]) / 2
print(f'Densest tile centre: x={best_x:.1f}, y={best_y:.1f} ({H.max():.0f} transcripts in that coarse tile)')
print('\nUse this as the window centre in psf_validation_breast.py instead of the whole-slide mean.')
