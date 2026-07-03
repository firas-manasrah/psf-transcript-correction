"""
PSF Wiener correction validation on GSE273557 Region 1 — PER-NEURON metric.

psf_validation_v2.py measures z-std per GENE across the entire 500x500um
window — but that window contains many neurons at genuinely different
depths (real biological z-spread of ~13-29um), which swamps a ~2-3um PSF
effect. That's very likely why it shows flat/negative results regardless
of argmax vs centroid: it's measuring the wrong thing at the wrong
spatial scale.

This script instead measures z-std of transcripts WITHIN each individual
annotated neuron's ~10um vicinity — the same spatial scale used in
check_oninstrument_correction.py, and the same logic as the simulation's
per-cell population metric (which is the one that actually showed real,
ground-truth-confirmed improvement). Apples-to-apples this time.
"""
import numpy as np
import pandas as pd
from scipy.fft import fftn, ifftn, fftfreq
from pathlib import Path

OUT      = Path('/home/jovyan/scratch/price_lab_drg')
PIXEL    = 0.2125
SIGMA_XY = 0.2
SIGMA_Z  = 2.8
VOXEL    = 0.5
RADIUS   = 10.0   # um — neuron soma scale, same as check_oninstrument_correction.py

print('Loading data...')
tx = pd.read_parquet(str(OUT / 'region1_transcripts.parquet'))
tx = tx[tx['qv'] >= 20].copy()
print(f'Transcripts QV>=20: {len(tx):,}')

meta = pd.read_csv(str(OUT / 'metadata.csv.gz'))
meta_r1 = meta[meta['Xenium_id'].str.contains('Region_1')].copy()
meta_r1['cx'] = meta_r1['cell_id'].str.split('-').str[0].astype(float) * PIXEL
meta_r1['cy'] = meta_r1['cell_id'].str.split('-').str[1].astype(float) * PIXEL
print(f'Annotated neurons Region 1: {len(meta_r1)}')

cx_mean = meta_r1['cx'].mean()
cy_mean = meta_r1['cy'].mean()
W = 500
x1, x2 = cx_mean - W / 2, cx_mean + W / 2
y1, y2 = cy_mean - W / 2, cy_mean + W / 2

tx_w = tx[(tx['x_location'] >= x1) & (tx['x_location'] <= x2) &
          (tx['y_location'] >= y1) & (tx['y_location'] <= y2)].copy()
neurons_w = meta_r1[(meta_r1['cx'] >= x1) & (meta_r1['cx'] <= x2) &
                     (meta_r1['cy'] >= y1) & (meta_r1['cy'] <= y2)]

print(f'Window {W}x{W}um: {len(tx_w):,} transcripts, {len(neurons_w)} neurons')

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

win_z = max(1, int(round(1.5 * SIGMA_Z / VOXEL)))
thresh_frac = 0.3

def centroid_correct_z(row):
    gx = int(np.clip((row['x_location'] - x1) / VOXEL, 0, NX - 1))
    gy = int(np.clip((row['y_location'] - y1) / VOXEL, 0, NY - 1))
    gz = int(np.clip((row['z_location'] - z_min) / VOXEL, 0, NZ - 1))
    z1i = max(0, gz - win_z)
    z2i = min(NZ, gz + win_z + 1)
    patch = corr[gx, gy, z1i:z2i].astype(float)
    if len(patch) == 0 or patch.max() < 1e-9:
        return row['z_location']
    thr = patch.max() * thresh_frac
    patch = np.where(patch >= thr, patch, 0.0)
    wsum = patch.sum()
    if wsum < 1e-9:
        return row['z_location']
    z_idx = np.arange(z1i, z2i)
    centroid_iz = (z_idx * patch).sum() / wsum
    return z_min + centroid_iz * VOXEL

print(f'\n=== PER-NEURON z-std reduction (radius={RADIUS}um, same scale as ===')
print(f'=== check_oninstrument_correction.py) ===')
print(f'{"neuron_id":10s} {"n_tx":6s} {"z_std_before":13s} {"z_std_after":12s} {"improvement":11s}')
print('-' * 60)

results = []
for idx, row in neurons_w.iterrows():
    dx = tx_w['x_location'] - row['cx']
    dy = tx_w['y_location'] - row['cy']
    dist = np.sqrt(dx**2 + dy**2)
    nearby = tx_w[dist <= RADIUS]
    if len(nearby) < 20:
        continue

    z_before = nearby['z_location'].values
    z_std_b = z_before.std()

    z_after = nearby.apply(centroid_correct_z, axis=1).values
    z_std_a = np.array(z_after).std()

    imp = (z_std_b - z_std_a) / z_std_b * 100
    print(f'{idx:10d} {len(nearby):6d} {z_std_b:13.3f} {z_std_a:12.3f} {imp:+10.1f}%')
    results.append({'neuron_id': idx, 'n': len(nearby),
                     'z_std_before': z_std_b, 'z_std_after': z_std_a,
                     'improvement_pct': imp})

results_df = pd.DataFrame(results)
print(f'\n=== SUMMARY across {len(results_df)} neurons ===')
print(f'Mean z_std before: {results_df.z_std_before.mean():.3f} um')
print(f'Mean z_std after:  {results_df.z_std_after.mean():.3f} um')
print(f'Mean per-neuron improvement: {results_df.improvement_pct.mean():+.1f}%')
print(f'Median per-neuron improvement: {results_df.improvement_pct.median():+.1f}%')
print(f'Neurons improved (>0%): {(results_df.improvement_pct > 0).sum()} / {len(results_df)}')

results_df.to_csv(str(OUT / 'psf_validation_per_neuron_results.csv'), index=False)
print(f"\nSaved to {OUT / 'psf_validation_per_neuron_results.csv'}")
print('Done.')
