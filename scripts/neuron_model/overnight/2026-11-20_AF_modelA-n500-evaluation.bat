@echo off
rem 2026-11-18, chain AF (GPU generation ~10-20 min, CPU Poisson ~0.5 h, evaluation ~3 h). Run after git pull; independent of AE (can run
rem in a second terminal, but both use the CPU).
rem
rem AD compared mogen_v3 (500 meshes) against mogen_A (200 meshes): different set sizes bias MMD / coverage. Here MoGen A gets 500 meshes too
rem (same seed 1, a NEW set seed_1_n500, the old 200 stay untouched; the first 200 clouds are the same noise as in the old set), then the
rem same evaluation as AD with equal sizes. One generation seed (as in AD); the memorization check by Chamfer is off (needs the cloud cache of
rem 680 534 train spines); the morphometric memorization check stays.
rem The checkpoint of the old A set is printed at the start (checkpoint_id in the sample json): CK below must be the same. Default of
rem generate_mogen.py is best_val_loss.
rem Log: scripts\neuron_model\overnight\2026-11-20_AF_modelA-n500-evaluation.bat > logs_AF.txt 2>&1

set CK=latest
set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set SPLITS=data\processed\neuron-model\splits
set RUNS=runs\training\neuron-model
set ANA=runs\analysis\neuron-model
set CFG=configs\neuron-model\reconstruction\poisson_sweep
set V3=%RUNS%\2026-11-03_015237_mogen-conditional-v3
set MA=%RUNS%\2026-10-20_213103_mogen-ft-dev
set MAN=%MA%\meshes_poisson_d8_pw8_n500
set CMP=%ANA%\2026-11-18_v3-vs-A
set FULL=C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/manifest_minnie65_full.parquet
set MORPH=C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/morphometrics_minnie65_full.parquet

cd /d %REPO%
if not exist %CMP%\real\poisson (echo NO REAL CONTROLS - run AB1 first & goto :end)
if not exist %V3%\meshes_poisson_d8_pw8\seed_1 (echo NO v3 MESHES & goto :end)

echo ===== [AF0] checkpoint of the existing A set (must equal CK=%CK%) =====
echo checkpoints in the run:
dir /b %MA%\checkpoints
for /f "delims=" %%F in ('dir /s /b %MA%\meshes\seed_1\*.json 2^>nul') do (findstr /i "checkpoint_id" "%%F" & goto :ckdone)
:ckdone

echo ===== [AF1] MoGen A: 300 more clouds (500 in total), seed 1, tag seed_1_n500 ===== %date% %time%
python scripts\neuron_model\generate_mogen.py --run %MA% --checkpoint %CK% --n-samples 500 --seed 1 --tag seed_1_n500 --no-mesh
if errorlevel 1 (echo GENERATION FAILED & goto :end)

echo ===== [AF2] Screened Poisson d8_pw8 for the 500 clouds ===== %date% %time%
python scripts\neuron_model\reconstruct_mogen_poisson.py --run %MA% --meshes-dir %MA%\meshes --set seed_1_n500 --out %MAN% --poisson-config %CFG%\d8_pw8.yaml --workers 8
if errorlevel 1 (echo RECONSTRUCTION FAILED & goto :end)
rem evaluate_generators.py reads <model dir>\seed_<s>: the new set gets that name inside its own folder
ren %MAN%\seed_1_n500 seed_1

echo ===== [AF3] evaluation v3 (500) vs A (500) vs real controls (300) ===== %date% %time%
python scripts\neuron_model\evaluate_generators.py ^
  --model mogen_v3=%V3%\meshes_poisson_d8_pw8 --model mogen_A=%MAN% --model real_poisson=%CMP%\real\poisson ^
  "data.manifests=['%FULL%']" data.path_mode=absolute data.splits=%SPLITS%\splits_v1_val_minus_dev.parquet ^
  data.bbox=%SPLITS%\bbox_splits_v1.json data.drop_unassigned=true "protocol.generation_seeds=[1]" real.morphometrics_table=%MORPH% ^
  real.chamfer_memorization=false name=generator-evaluation-v3-vs-A-n500
if errorlevel 1 (echo EVALUATION FAILED & goto :end)
for /f "delims=" %%d in ('dir /b /ad /o-n %ANA%\*_generator-evaluation-v3-vs-A-n500') do (set EVAL=%ANA%\%%d& goto :evalfound)
:evalfound
echo evaluation run: %EVAL%
python scripts\neuron_model\classifier_ablation.py --eval-run %EVAL% --reference real_poisson --models mogen_v3 mogen_A
python scripts\neuron_model\morphometric_tails.py --eval-run %EVAL% --metrics Volume Area Length ConvexHullVolume
type %EVAL%\postprocessed\summary.md

:end
echo ===== chain AF done ===== %date% %time%
