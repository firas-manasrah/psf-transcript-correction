"""
Empirical check: is the real z_location scatter in Xenium output already
much smaller than the theoretical raw PSF sigma_z=2.8um? If so, that's
direct evidence 10x's on-instrument PSF calibration (documented in their
Image Processing Algorithms page: "instrument-specific calibrations...
adjust the optical system's PSF model" and "XYZ coordinates of each
punctum are refined by examining local brightness") already did most of
the correctable work before this file was ever written — which would
explain why a second-pass Wiener correction on the density grid finds
little residual error to fix.

Method: look at z-scatter of transcripts WITHIN a tightly compact,
biologically real structure (a single annotated neuron's immediate
vicinity) rather than across a whole gene population (which mixes real
biological spread with any instrument noise). A true neuron soma is a
few um across; if raw z_location scatter within one neuron already
approaches or beats 2.8um, on-instrument correction is doing real work.
"""
import numpy as np
import pandas as pd
from pathlib import Path

OUT   = Path('/home/jovyan/scratch/price_lab_drg')
PIXEL = 0.2125
SIGMA_Z_THEORETICAL = 2.8   # um, raw PSF sigma_z if uncorrected

print('Loading data...')
tx = pd.read_parquet(str(OUT / 'region1_transcripts.parquet'))
tx = tx[tx['qv'] >= 20].copy()

meta = pd.read_csv(str(OUT / 'metadata.csv.gz'))
meta_r1 = meta[meta['Xenium_id'].str.contains('Region_1')].copy()
meta_r1['cx'] = meta_r1['cell_id'].str.split('-').str[0].astype(float) * PIXEL
meta_r1['cy'] = meta_r1['cell_id'].str.split('-').str[1].astype(float) * PIXEL

print(f'Annotated neurons: {len(meta_r1)}')
print(f'Theoretical raw PSF sigma_z: {SIGMA_Z_THEORETICAL} um\n')

# For each annotated neuron, take all transcripts within a tight 10um
# radius (neuron soma scale) and measure z-scatter directly.
radius = 10.0  # um — typical DRG neuron soma radius, tight enough that
                # z-spread reflects instrument/PSF noise, not biology
                # spanning multiple structures

print(f'{"neuron_id":10s} {"n_tx":6s} {"z_std_um":10s} {"vs_raw_PSF":12s}')
print('-' * 45)

z_stds = []
for idx, row in meta_r1.iterrows():
    dx = tx['x_location'] - row['cx']
    dy = tx['y_location'] - row['cy']
    dist = np.sqrt(dx**2 + dy**2)
    nearby = tx[dist <= radius]
    if len(nearby) < 20:
        continue
    z_std = nearby['z_location'].std()
    z_stds.append(z_std)
    ratio = z_std / SIGMA_Z_THEORETICAL
    print(f'{idx:10d} {len(nearby):6d} {z_std:10.3f} {ratio:11.2f}x')

z_stds = np.array(z_stds)
print(f'\n=== SUMMARY across {len(z_stds)} neurons ===')
print(f'Mean observed z-std within neuron soma: {z_stds.mean():.3f} um')
print(f'Theoretical raw PSF sigma_z:            {SIGMA_Z_THEORETICAL} um')
print(f'Ratio (observed / theoretical):         {z_stds.mean()/SIGMA_Z_THEORETICAL:.2f}x')
print()
if z_stds.mean() < SIGMA_Z_THEORETICAL * 0.6:
    print('=> Observed z-scatter is substantially BELOW the raw theoretical')
    print('   PSF sigma. Strong evidence that on-instrument PSF calibration')
    print('   already removed most of the correctable axial error before')
    print('   this file was written. A second-pass correction has little')
    print('   residual error left to find — consistent with the flat/')
    print('   negative results seen in psf_validation_v2.py on real data.')
elif z_stds.mean() < SIGMA_Z_THEORETICAL * 0.9:
    print('=> Observed z-scatter is moderately below the raw theoretical')
    print('   PSF sigma. Some on-instrument correction likely happened,')
    print('   but meaningful residual error may still remain.')
else:
    print('=> Observed z-scatter is close to the raw theoretical PSF sigma.')
    print('   Little evidence of on-instrument correction — the flat/')
    print('   negative validation results likely have a different cause')
    print('   (e.g. correction method itself, window sizing, grid resolution).')
