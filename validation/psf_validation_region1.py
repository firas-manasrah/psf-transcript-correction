"""
PSF Wiener correction validation on GSE273557 Region 1
Yu et al. Nature Neuroscience 2024 — Human DRG Xenium
1,415 manually annotated neurons as ground truth
"""
import numpy as np
import pandas as pd
from scipy.fft import fftn, ifftn, fftfreq
from scipy.ndimage import gaussian_filter
from pathlib import Path

OUT = Path('/home/jovyan/scratch/price_lab_drg')

# PSF parameters — Xenium instrument
SIGMA_XY_UM = 0.2
SIGMA_Z_UM  = 2.8
VOXEL       = 0.5  # coarser voxel for speed on large dataset

print('Loading Region 1 transcripts...')
tx = pd.read_parquet(str(OUT/'region1_transcripts.parquet'))
tx = tx[tx['qv'] >= 20].copy()
print(f'QV>=20: {len(tx):,} transcripts')

# Focus on boundary transcripts — near manually annotated neurons
# Load metadata to get neuron positions
meta = pd.read_csv(str(OUT/'metadata.csv.gz'))
# Filter to Region 1 (Xenium_id contains Region_1)
meta_r1 = meta[meta['Xenium_id'].str.contains('Region_1')]
print(f'Manually annotated neurons in Region 1: {len(meta_r1)}')

# Parse neuron centroids from cell_id (format: x-y)
meta_r1 = meta_r1.copy()
meta_r1['cx'] = meta_r1['cell_id'].str.split('-').str[0].astype(float)
meta_r1['cy'] = meta_r1['cell_id'].str.split('-').str[1].astype(float)
print(f'Neuron X range: {meta_r1.cx.min():.0f} to {meta_r1.cx.max():.0f}')
print(f'Neuron Y range: {meta_r1.cy.min():.0f} to {meta_r1.cy.max():.0f}')

# Focus on a 500x500 um window around the densest neuron area
# Find centre of mass of neurons
cx_mean = meta_r1['cx'].mean()
cy_mean = meta_r1['cy'].mean()
print(f'Neuron centre of mass: ({cx_mean:.0f}, {cy_mean:.0f})')

# Define analysis window
W = 500  # um
x1 = cx_mean - W/2; x2 = cx_mean + W/2
y1 = cy_mean - W/2; y2 = cy_mean + W/2

tx_w = tx[(tx['x_location']>=x1)&(tx['x_location']<=x2)&
          (tx['y_location']>=y1)&(tx['y_location']<=y2)].copy()
print(f'\nWindow {W}x{W} um around neuron centre:')
print(f'  Transcripts: {len(tx_w):,}')
print(f'  Z range: {tx_w.z_location.min():.2f} to {tx_w.z_location.max():.2f} um')

neurons_w = meta_r1[
    (meta_r1['cx']>=x1)&(meta_r1['cx']<=x2)&
    (meta_r1['cy']>=y1)&(meta_r1['cy']<=y2)]
print(f'  Annotated neurons: {len(neurons_w)}')

# Build 3D transcript density field
NX = int((x2-x1)/VOXEL)+2
NY = int((y2-y1)/VOXEL)+2
NZ = int((tx_w.z_location.max()-tx_w.z_location.min())/VOXEL)+2
print(f'\nDensity grid: {NX}x{NY}x{NZ} voxels at {VOXEL}um resolution')

ix = ((tx_w['x_location']-x1)/VOXEL).astype(int).clip(0,NX-1)
iy = ((tx_w['y_location']-y1)/VOXEL).astype(int).clip(0,NY-1)
z_min = tx_w['z_location'].min()
iz = ((tx_w['z_location']-z_min)/VOXEL).astype(int).clip(0,NZ-1)

density = np.zeros((NX,NY,NZ), dtype=np.float32)
for x,y,z in zip(ix,iy,iz):
    density[x,y,z] += 1

print(f'Density field max: {density.max():.0f} transcripts/voxel')

# Apply Wiener PSF correction
print('\nApplying Wiener PSF correction...')
SIGMA_XY = SIGMA_XY_UM / VOXEL
SIGMA_Z  = SIGMA_Z_UM  / VOXEL

kx,ky,kz = fftfreq(NX),fftfreq(NY),fftfreq(NZ)
KX,KY,KZ = np.meshgrid(kx,ky,kz,indexing='ij')
OTF = np.exp(-2*np.pi**2*(SIGMA_XY**2*(KX**2+KY**2)+SIGMA_Z**2*KZ**2))
lambda_reg = (density.std()*0.04)**2 / max(density.var(), 1e-9)
lambda_reg = float(np.clip(lambda_reg, 1e-5, 0.1))
print(f'Lambda: {lambda_reg:.5f}')

wiener = OTF / (OTF**2 + lambda_reg)
corrected = np.real(ifftn(fftn(density.astype(np.float64)) * wiener)).clip(0)
print(f'Correction complete')

# Validate — for each annotated neuron compute:
# 1. How many NEFH transcripts are within the neuron boundary before/after
# 2. Z-position distribution of NEFH transcripts before/after

print('\n=== PSF VALIDATION ===')
# Use NEFH as large neuron marker — should be concentrated in large neurons
for gene in ['NEFH', 'TRPV1', 'PIEZO2', 'TRPM8']:
    gene_tx = tx_w[tx_w['feature_name']==gene]
    if len(gene_tx) == 0:
        print(f'{gene}: not found in window')
        continue

    # Z distribution before correction
    z_before = gene_tx['z_location'].values
    z_mean_before = z_before.mean()
    z_std_before  = z_before.std()

    # Find corrected z for each transcript
    # Local maximum in corrected density within 1 sigma_z
    win_z = max(1, int(SIGMA_Z_UM/VOXEL))
    z_corrected = []
    for _,row in gene_tx.iterrows():
        gx = int((row['x_location']-x1)/VOXEL)
        gy = int((row['y_location']-y1)/VOXEL)
        gz = int((row['z_location']-z_min)/VOXEL)
        gx = np.clip(gx,0,NX-1)
        gy = np.clip(gy,0,NY-1)
        gz = np.clip(gz,0,NZ-1)
        z1i = max(0,gz-win_z); z2i = min(NZ,gz+win_z+1)
        patch = corrected[gx,gy,z1i:z2i]
        if len(patch)==0 or patch.max()<1e-9:
            z_corrected.append(row['z_location'])
            continue
        peak_iz = z1i + patch.argmax()
        z_corrected.append(z_min + peak_iz*VOXEL)

    z_after = np.array(z_corrected)
    z_mean_after = z_after.mean()
    z_std_after  = z_after.std()
    improvement  = (z_std_before - z_std_after) / z_std_before * 100

    print(f'{gene:10s}: n={len(gene_tx):4d}  '
          f'z_std before={z_std_before:.3f}um  '
          f'after={z_std_after:.3f}um  '
          f'improvement={improvement:.1f}%')

print('\nDone.')
