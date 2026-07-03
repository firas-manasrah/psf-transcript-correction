#!/bin/bash
# Download 10x Genomics Xenium FFPE Human Breast Cancer Rep1 (Janesick et al. 2023)
mkdir -p ~/scratch/xenium_breast && cd ~/scratch/xenium_breast
curl -O https://cf.10xgenomics.com/samples/xenium/1.0.1/Xenium_FFPE_Human_Breast_Cancer_Rep1/Xenium_FFPE_Human_Breast_Cancer_Rep1_outs.zip
unzip Xenium_FFPE_Human_Breast_Cancer_Rep1_outs.zip
ls -lh
