@echo off
setlocal enabledelayedexpansion
rem 2026-11-27, chain AT: two HTML pages for looking at the current generators. CPU (the atlas encodes with the VAE: GPU if free), ~1-2 h,
rem reads the NAS (real branch meshes, real spine meshes). No training, nothing is overwritten.
rem AT1: assembly of the current T+P set (T0 with the neuron effect, mode per_neuron, chain AN; P1b2) with the spines of MoGen A
rem      (d8_pw8, 500 meshes), 80 branches -> <set>\assembly_at (+ evaluation)
rem AT2: view_generated_branches.py: 40 assembled branches next to their real source branches (real mesh from the raw segments
rem      on the NAS), radius along the branch, spine positions, nearest neighbours, metrics -> <set>\assembly_at\view_generated_branches.html
rem AT3: atlas of the VAE latent space: real spines of the clean evaluation neurons (val v1 minus dev, as slides 13.3),
rem      MoGen A (500) and VAE_flow (model C), a mesh for EVERY point, 3 nearest spines of every source under the selected one
rem      -> runs\analysis\neuron-model\atlas\2026-11-27_at\spine_atlas.html
rem Log: scripts\neuron_model\overnight\2026-11-27_AT_branch-viewer-and-atlas.bat > logs_AT.txt 2>&1

set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set B=C:\Users\Student\LipsEkaterina\data\minnie65\preprocessed_branches
set RAW=O:\Datasets
set RUNS=runs\training\neuron-model
set ANA=runs\analysis\neuron-model
set SPLITS=data\processed\neuron-model\splits
set TPN=%ANA%\tp_generation\minnie_h01_neuron\minnie65\generated
set MA=%RUNS%\2026-10-20_213103_mogen-ft-dev
set MAN=%MA%\meshes_poisson_d8_pw8_n500\seed_1
set VAE=%RUNS%\2026-10-19_003408_vae-dev-reg0-cos
set FULL=C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/manifest_minnie65_full.parquet

cd /d %REPO%

set TP=
for /f "delims=" %%d in ('dir /b /ad /o-n %TPN% ^| findstr /i "t0n-per_neuron_seed1"') do (set TP=%TPN%\%%d& goto :tpfound)
:tpfound
if "%TP%"=="" (echo NO per_neuron set of chain AN in %TPN% & goto :at3)
if not exist %MAN%\sample_00000 (echo NO MoGen A meshes %MAN% & goto :at3)
echo T+P set: %TP%

echo ===== [AT1] assembly with MoGen A spines ===== %date% %time%
if exist %TP%\assembly_at\assembly_branches.parquet (echo assembly exists, skipped) else (
  python scripts\neuron_model\assemble_branches.py --tp-dir %TP% --spines %MAN% --context-spines %B%\branch_spines.parquet ^
    --n-branches 80 --seed 1 --out %TP%\assembly_at
  if errorlevel 1 (echo ASSEMBLY FAILED & goto :at3)
  python scripts\neuron_model\evaluate_assembled_branches.py --tp-dir %TP% --assembly %TP%\assembly_at --out %TP%\assembly_at\evaluation
)

echo ===== [AT2] real vs generated branches page ===== %date% %time%
python scripts\neuron_model\view_generated_branches.py --tp-dir %TP% --assembly %TP%\assembly_at --raw-data-root %RAW% --dataset minnie65 --n 40
if errorlevel 1 (echo BRANCH PAGE FAILED)

:at3
echo ===== [AT3] atlas of the VAE latent space: real / MoGen A / VAE_flow ===== %date% %time%
set VF=
for /f "delims=" %%d in ('dir /b /ad %VAE%\meshes_flow_prior 2^>nul') do (if exist %VAE%\meshes_flow_prior\%%d\sample_00000 if "!VF!"=="" set VF=%VAE%\meshes_flow_prior\%%d)
if "%VF%"=="" if exist %VAE%\meshes_flow_prior\sample_00000 set VF=%VAE%\meshes_flow_prior
if "%VF%"=="" (echo NO VAE_flow meshes in %VAE%\meshes_flow_prior & goto :end)
echo VAE_flow meshes: %VF%
python scripts\neuron_model\build_spine_atlas.py --vae-run %VAE% --manifest %FULL% --path-mode absolute ^
  --splits %SPLITS%\splits_v1_val_minus_dev.parquet --split test --n-real 600 --real-meshes 600 --n-train 1000 ^
  --model MoGen_A=%MAN% --model VAE_flow=%VF% --n-gen 500 --gen-meshes 500 --max-faces 1200 --neighbors 3 ^
  --out %ANA%\atlas\2026-11-27_at\spine_atlas.html
if errorlevel 1 (echo ATLAS FAILED)

:end
echo ===== chain AT done ===== %date% %time%
