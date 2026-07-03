"""
PSF Wiener correction validation on 10x Genomics Xenium FFPE Human Breast
Cancer Rep1 (Janesick et al. 2023, Nature Communications).
Public dataset — no permission needed, CC-licensed via 10x Genomics.

Second-dataset generalization test for psf-transcript-correction:
same platform (Xenium) as the DRG validation, different tissue
architecture (dense epithelial/ductal breast vs ECM-rich DRG neuropil).

Download first (run on cluster):
    bash download_breast.sh
"""
import numpy as np
import pandas as pd
from scipy.fft import fftn, ifftn, fftfreq
from pathlib import Path

OUT      = Path('/home/jovyan/scratch/xenium_breast/outs')
SIGMA_XY = 0.2      # um — same optical PSF as DRG validation
SIGMA_Z  = 2.8      # um
VOXEL    = 0.5      # um per voxel for density grid

# Breast marker genes spanning distinct compartments — a correction
# that only helps DRG neuronal markers is a much weaker generalization
# claim than one that also helps genes with clean, restricted expression
# domains in dense epithelial tissue.
GENES = {
    'EPCAM':  'epithelial (broad)',
    'ERBB2':  'tumor biomarker (HER2)',
    'ESR1':   'tumor biomarker (ER)',
    'KRT14':  'myoepithelial',
    'ACTA2':  'myoepithelial/smooth muscle',
    'PTPRC':  'immune (CD45)',
    'PECAM1': 'endothelial',
}

print('Loading transcripts...')
tx = pd.read_parquet(str(OUT / 'transcripts.parquet'))
tx = tx[tx['qv'] >= 20].copy()
# BUG FIX: feature_name is stored as bytes, not str — decode before any
# string comparison, or every gene lookup silently returns 0 rows.
if len(tx) > 0 and isinstance(tx['feature_name'].iloc[0], bytes):
    tx['feature_name'] = tx['feature_name'].str.decode('utf-8')
print(f'Transcripts QV>=20: {len(tx):,}')
print(f'Z range: {tx.z_location.min():.2f} to {tx.z_location.max():.2f} um')

# Window centred on a known dense-tissue tile — the whole-slide transcript
# mean was tested and landed in sparse space (0 marker transcripts in a
# 144K-transcript window). Centre confirmed by diagnose_breast.py:
# densest 500x500um-scale tile had 249,187 transcripts.
cx_mean = 7324.7
cy_mean = 2163.2
W = 500
x1, x2 = cx_mean - W / 2, cx_mean + W / 2
y1, y2 = cy_mean - W / 2, cy_mean + W / 2

tx_w = tx[(tx['x_location'] >= x1) & (tx['x_location'] <= x2) &
          (tx['y_location'] >= y1) & (tx['y_location'] <= y2)].copy()
print(f'Window {W}x{W}um: {len(tx_w):,} transcripts')

# Build 3D density grid
z_min = tx_w['z_location'].min()
NX = int((x2 - x1) / VOXEL) + 2
NY = int((y2 - y1) / VOXEL) + 2
NZ = int((tx_w['z_location'].max() - z_min) / VOXEL) + 2
print(f'Density grid: {NX}x{NY}x{NZ} voxels')

ix = ((tx_w['x_location'] - x1) / VOXEL).astype(int).clip(0, NX - 1).values
iy = ((tx_w['y_location'] - y1) / VOXEL).astype(int).clip(0, NY - 1).values
iz = ((tx_w['z_location'] - z_min) / VOXEL).astype(int).clip(0, NZ - 1).values

density = np.zeros((NX, NY, NZ), dtype=np.float32)
np.add.at(density, (ix, iy, iz), 1)
print(f'Density max: {density.max():.0f}')

# Wiener correction — identical method/params to the DRG validation
print('Applying Wiener correction...')
sx = SIGMA_XY / VOXEL
sz = SIGMA_Z / VOXEL
kx, ky, kz = fftfreq(NX), fftfreq(NY), fftfreq(NZ)
KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
OTF = np.exp(-2 * np.pi**2 * (sx**2 * (KX**2 + KY**2) + sz**2 * KZ**2))
lam = float(np.clip((density.std() * 0.04)**2 / max(density.var(), 1e-9), 1e-5, 0.1))
W_f = OTF / (OTF**2 + lam)
corr = np.real(ifftn(fftn(density.astype(np.float64)) * W_f)).clip(0)
print(f'Lambda={lam:.5f}  corrected max={corr.max():.1f}')

# Validation: z-position scatter before/after, per marker gene
print('\n=== VALIDATION RESULTS (breast, Rep1) ===')
print(f'{"Gene":8s} {"compartment":28s} {"n":6s} {"z_std_before":12s} {"z_std_after":11s} {"improvement":11s}')
print('-' * 85)

results = []

for gene, compartment in GENES.items():
    g_tx = tx_w[tx_w['feature_name'] == gene]
    if len(g_tx) < 10:
        print(f'{gene:8s} {compartment:28s} skipped (n<10, n={len(g_tx)})')
        continue

    z_before = g_tx['z_location'].values
    z_std_b = z_before.std()

    # Corrected z via weighted centroid (NOT hard argmax) — matches
    # psf_simulation_paper1.py and psf_validation_v2.py. Argmax is highly
    # sensitive to single noisy voxels from Wiener deconvolution's
    # high-frequency noise amplification; a local weighted centroid is
    # far more robust. win_z sized to ~1.5x PSF sigma_z.
    win_z = max(1, int(round(1.5 * SIGMA_Z / VOXEL)))
    thresh_frac = 0.3
    z_after = []
    for _, row in g_tx.iterrows():
        gx = int(np.clip((row['x_location'] - x1) / VOXEL, 0, NX - 1))
        gy = int(np.clip((row['y_location'] - y1) / VOXEL, 0, NY - 1))
        gz = int(np.clip((row['z_location'] - z_min) / VOXEL, 0, NZ - 1))
        z1i = max(0, gz - win_z)
        z2i = min(NZ, gz + win_z + 1)
        patch = corr[gx, gy, z1i:z2i].astype(float)
        if len(patch) == 0 or patch.max() < 1e-9:
            z_after.append(row['z_location'])
            continue
        thr = patch.max() * thresh_frac
        patch = np.where(patch >= thr, patch, 0.0)
        wsum = patch.sum()
        if wsum < 1e-9:
            z_after.append(row['z_location'])
            continue
        z_idx = np.arange(z1i, z2i)
        centroid_iz = (z_idx * patch).sum() / wsum
        z_after.append(z_min + centroid_iz * VOXEL)

    z_std_a = np.array(z_after).std()
    imp = (z_std_b - z_std_a) / z_std_b * 100

    print(f'{gene:8s} {compartment:28s} {len(g_tx):6d} {z_std_b:12.3f} {z_std_a:11.3f} {imp:+10.1f}%')
    results.append({'gene': gene, 'compartment': compartment, 'n': len(g_tx),
                     'z_std_before': z_std_b, 'z_std_after': z_std_a,
                     'improvement_pct': imp})

pd.DataFrame(results).to_csv(str(OUT / 'psf_validation_breast_results.csv'), index=False)
print(f"\nSaved results to {OUT / 'psf_validation_breast_results.csv'}")
print('Done.')
