@echo off
rem Overnight 2026-11-01, chain B (NAS reads of Minnie65, then the GPU). Run from the repository root, env neuron-model active.
rem   1. local cache of splits_v1 for S: 60 000 train + 3 000 val spines (point clouds variants 1-2 + SDF, ~3 h, ~50 GB)
rem   2. conditional MoGen A on splits_v1, started from the pretrained mouse_mixed weights (not from the dev-trained A:
rem      16 of its training neurons are in val/test of v1), block cell_type_binary, 30 000 steps (~5 h on the RTX 4090).
rem      Trains on whatever part of the cache is complete (the cache manifest lists complete spines only).
rem Chain A (2026-11-01_A_h01-s-v2_branches.bat) can run at the same time in another window.

set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set CACHE=%REPO%\cache\minnie65_v1_60k
set SPLITS=data\processed\neuron-model\splits\splits_v1.parquet

cd /d %REPO%

echo ===== [B1/2] cache of splits_v1 -^> %CACHE% ===== %date% %time%
python scripts\neuron_model\cache_subset.py --manifest O:\Datasets\Minnie65\preprocessed\manifest.parquet --splits %SPLITS% ^
  --out %CACHE% --n-train 60000 --n-val 3000 --variants 1 2 --threads 32

echo ===== [B2/2] conditional MoGen A on splits_v1 (from mouse_mixed) ===== %date% %time%
python scripts\neuron_model\train_mogen.py --config configs\neuron-model\mogen\conditional_finetune.yaml ^
  model.init=pretrained model.pretrained_weights=data/external/mogen/mouse_mixed_750000_weights.npz model.pretrained_collection=ema_params ^
  "data.manifests=['%CACHE:\=/%/manifest.parquet']" data.splits=%SPLITS:\=/% data.drop_unassigned=true "data.variants=[1,2]" "data.check_files=[]" ^
  "condition.blocks=[cell_type_binary]" train.max_steps=30000 train.validate_every=30000 name=mogen-conditional-v1

echo ===== B done ===== %date% %time%
