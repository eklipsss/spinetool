@echo off
rem Overnight 2026-11-07, chain K (GPU, ~3-4 h): VAE generation from a prior fitted on DIRECTLY OPTIMISED codes.
rem Evidence (decoder study 2026-11-07_195048, 300 dev-val spines): the frozen decoder with the code of each spine fitted to its
rem SDF samples has almost no bias (mean |bias| 0.070 vs 0.226 for z = mu; ConvexHullRatio +0.06 vs -0.59; Chamfer 18 vs 40 nm;
rem below the 0.161 of the real mesh itself through exact SDF + MC). The necks are lost by the ENCODER (amortisation gap), the
rem decoder can draw them. A generator needs no encoder (prior -> code -> decoder), so the flow prior is fitted on optimised codes.
rem Same decoder (2026-10-19_003408_vae-dev-reg0-cos), same flow settings as latent_prior (8 layers, 256, 300 epochs), same
rem 200 samples / seed 1 / MC 128^3; ONE evaluation on the untouched test with the old flow set and the real_mc control.
rem Log: scripts\neuron_model\overnight\2026-11-07_K_vae-optimised-codes.bat > logs_K.txt 2>&1

set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set VAE=runs\training\neuron-model\2026-10-19_003408_vae-dev-reg0-cos
set CTRL=runs\analysis\neuron-model\2026-11-06_004118_real-controls
set FULL=C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/manifest_minnie65_full.parquet
set MORPH=C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/morphometrics_minnie65_full.parquet

cd /d %REPO%

echo ===== [K1/5] optimised codes of 15 000 train + 1 500 val spines ===== %date% %time%
python scripts\neuron_model\fit_vae_codes.py --run %VAE% --n-train 15000 --n-val 1500 --steps 500 --batch-size 32
if errorlevel 1 goto :end

echo ===== [K2/5] flow prior on the optimised codes ===== %date% %time%
python scripts\neuron_model\fit_latent_prior.py --run %VAE% --variants 1 --out-name latent_prior_zopt --n-train 15000 --n-val 1500 --gmm 4 8
if errorlevel 1 goto :end

echo ===== [K3/5] 200 meshes from the new flow, MC 128^3 ===== %date% %time%
python scripts\neuron_model\generate_vae.py --run %VAE% --prior learned --latent-prior %VAE%\latent_prior_zopt\flow_prior.pt ^
  --n-samples 200 --seed 1 --resolution 128 --out-name meshes_zopt_flow_prior
if errorlevel 1 goto :end

echo ===== [K4/5] evaluation on the untouched test ===== %date% %time%
python scripts\neuron_model\evaluate_generators.py ^
  --model VAE_flow=%VAE%\meshes_flow_prior --model VAE_zopt_flow=%VAE%\meshes_zopt_flow_prior --model real_mc=%CTRL%\mc ^
  "data.manifests=['%FULL%']" data.path_mode=absolute ^
  data.splits=data/processed/neuron-model/splits/splits_dev_100n_untouched_test.parquet ^
  data.bbox=data/processed/neuron-model/splits/bbox_splits_dev_100n.json ^
  data.drop_unassigned=true "protocol.generation_seeds=[1]" real.morphometrics_table=%MORPH% name=generator-evaluation-zopt

echo ===== [K5/5] classifier ablation ===== %date% %time%
for /f "delims=" %%d in ('dir /b /ad /o-n runs\analysis\neuron-model\*_generator-evaluation-zopt') do (set EVAL=runs\analysis\neuron-model\%%d& goto :found)
:found
echo evaluation folder: %EVAL%
python scripts\neuron_model\classifier_ablation.py --eval-run %EVAL%
python scripts\neuron_model\classifier_ablation.py --eval-run %EVAL% --reference real_mc --models VAE_flow VAE_zopt_flow

:end
echo ===== K done ===== %date% %time%
