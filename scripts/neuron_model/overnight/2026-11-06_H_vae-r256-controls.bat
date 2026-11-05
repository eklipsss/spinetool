@echo off
rem Overnight 2026-11-06, chain H (GPU + CPU; can run next to the evaluation of the 2026-11-06 real controls).
rem Question: how much of the VAE gap (convexity, attachment metrics) is the 128^3 Marching-Cubes grid?
rem   1. the SAME 200 flow-prior latents (same prior file, same seed) decoded on the final 256^3 grid -> meshes_flow_prior_r256
rem   2. the SAME 200 real test spines as the mc control of 2026-11-06_004118 (same base seed) through exact SDF + MC 256^3
rem   3. evaluate_generators on the untouched-test split: VAE_flow 128 / 256 next to real_mc 128 / 256 and real_asis
rem   4. classifier ablation: VAE 256 against real spines and against real spines through the same 256^3 chain
rem Log: scripts\neuron_model\overnight\2026-11-06_H_vae-r256-controls.bat > logs_H.txt 2>&1

set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set VAE=runs\training\neuron-model\2026-10-19_003408_vae-dev-reg0-cos
set CTRL=runs\analysis\neuron-model\2026-11-06_004118_real-controls
set CTRL256=runs\analysis\neuron-model\2026-11-06_real-controls-r256
set SPLIT=data\processed\neuron-model\splits\splits_dev_100n_untouched_test.parquet
set FULL=C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/manifest_minnie65_full.parquet
set MORPH=C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/morphometrics_minnie65_full.parquet

cd /d %REPO%

echo ===== [H1/4] VAE flow prior, 256^3 ===== %date% %time%
python scripts\neuron_model\generate_vae.py --run %VAE% --prior learned --latent-prior %VAE%\latent_prior\flow_prior.pt --n-samples 200 --seed 1 --resolution 256 --out-name meshes_flow_prior_r256

echo ===== [H2/4] real mc control, 256^3 ===== %date% %time%
python scripts\neuron_model\make_real_controls.py --manifest cache\manifest_minnie65_meta.parquet --path-mode absolute --splits %SPLIT% --vae-run %VAE% --controls mc --resolution 256 --n-samples 200 --seeds 1 --workers 8 --out %CTRL256%

echo ===== [H3/4] evaluation ===== %date% %time%
python scripts\neuron_model\evaluate_generators.py ^
  --model VAE_flow=%VAE%\meshes_flow_prior --model VAE_flow_r256=%VAE%\meshes_flow_prior_r256 ^
  --model real_mc=%CTRL%\mc --model real_mc_r256=%CTRL256%\mc --model real_asis=%CTRL%\asis ^
  "data.manifests=['%FULL%']" data.path_mode=absolute ^
  data.splits=%SPLIT:\=/% data.bbox=data/processed/neuron-model/splits/bbox_splits_dev_100n.json ^
  data.drop_unassigned=true "protocol.generation_seeds=[1]" real.morphometrics_table=%MORPH% name=generator-evaluation-r256

echo ===== [H4/4] classifier ablation ===== %date% %time%
for /f "delims=" %%d in ('dir /b /ad /o-n runs\analysis\neuron-model\*_generator-evaluation-r256') do (set EVAL=runs\analysis\neuron-model\%%d& goto :found)
:found
echo evaluation folder: %EVAL%
python scripts\neuron_model\classifier_ablation.py --eval-run %EVAL%
python scripts\neuron_model\classifier_ablation.py --eval-run %EVAL% --reference real_mc_r256 --models VAE_flow_r256
python scripts\neuron_model\classifier_ablation.py --eval-run %EVAL% --reference real_mc --models VAE_flow
python scripts\neuron_model\classifier_ablation.py --eval-run %EVAL% --reference real_asis --models real_mc real_mc_r256

echo ===== H done ===== %date% %time%
