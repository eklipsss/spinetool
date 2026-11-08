@echo off
rem Overnight 2026-11-08, chain L (GPU ~5-6 h): VAE prior on codes fitted with the posterior anchor, SELECTED ON DEV VAL.
rem Evidence:
rem   chain K (untouched test, 2026-11-08_011445): flow on codes fitted WITHOUT a penalty breaks the neck in 43 % of the meshes
rem     (single-component meshes without a head, multi-component ones with a detached head), morphometric spread x3.1;
rem     not the MC grid (256^3: 0.58 single-component vs 0.57).
rem   decoder study 2026-11-08_015641 (300 dev-val spines): unpenalised codes sit at ((z-mu)/sigma)^2 = 13.6 (the decoder was
rem     trained at ~1); anchor weight 1 nm keeps them at 1.54 with almost the whole gain (mean |bias| 0.091 vs 0.087, z = mu
rem     0.238; ConvexHullRatio -0.16 n.s. vs -0.60; Chamfer 20.9 vs 17.7 nm). Midpoints of code pairs did NOT separate the
rem     variants (0.81-0.86 single component for all, mu included), so the tails of the prior are tested directly here.
rem The untouched test was used in chains I and K: everything here is compared on the dev VAL neurons (real.test_split=val,
rem with a real_mc control of val spines); the test is touched once more only for the final variant.
rem Log: scripts\neuron_model\overnight\2026-11-08_L_vae-anchored-codes.bat > logs_L.txt 2>&1

set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set VAE=runs\training\neuron-model\2026-10-19_003408_vae-dev-reg0-cos
set SPLIT=data\processed\neuron-model\splits\splits_dev_100n_untouched_test.parquet
set CTRLV=runs\analysis\neuron-model\2026-11-08_real-controls-val
set FULL=C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/manifest_minnie65_full.parquet
set MORPH=C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/morphometrics_minnie65_full.parquet

cd /d %REPO%

echo ===== [L1/7] anchored codes (weight 1 nm) of 15 000 train + 1 500 val spines ===== %date% %time%
python scripts\neuron_model\fit_vae_codes.py --run %VAE% --n-train 15000 --n-val 1500 --steps 500 --batch-size 32 --anchor-weight 1 --out-name latent_prior_zopt_a1
if errorlevel 1 goto :end

echo ===== [L2/7] flow prior on the anchored codes ===== %date% %time%
python scripts\neuron_model\fit_latent_prior.py --run %VAE% --variants 1 --out-name latent_prior_zopt_a1 --n-train 15000 --n-val 1500 --gmm 4 8
if errorlevel 1 goto :end

echo ===== [L3/7] 200 meshes from the anchored flow, MC 128^3 ===== %date% %time%
python scripts\neuron_model\generate_vae.py --run %VAE% --prior learned --latent-prior %VAE%\latent_prior_zopt_a1\flow_prior.pt ^
  --n-samples 200 --seed 1 --resolution 128 --out-name meshes_zopt_a1_flow_prior
if errorlevel 1 goto :end

echo ===== [L4/7] tails check: unpenalised flow at temperature 0.85 ===== %date% %time%
python scripts\neuron_model\generate_vae.py --run %VAE% --prior learned --latent-prior %VAE%\latent_prior_zopt\flow_prior.pt ^
  --n-samples 200 --seed 1 --resolution 128 --temperature 0.85 --out-name meshes_zopt_flow_prior_T085

echo ===== [L5/7] real mc control of 200 dev-val spines ===== %date% %time%
python scripts\neuron_model\make_real_controls.py --manifest cache\manifest_minnie65_meta.parquet --path-mode absolute --splits %SPLIT% --split val ^
  --vae-run %VAE% --controls mc --resolution 128 --n-samples 200 --seeds 1 --workers 8 --out %CTRLV%

echo ===== [L6/7] evaluation on the dev VAL neurons ===== %date% %time%
python scripts\neuron_model\evaluate_generators.py ^
  --model VAE_flow=%VAE%\meshes_flow_prior --model VAE_zopt_flow=%VAE%\meshes_zopt_flow_prior ^
  --model VAE_zopt_flow_T085=%VAE%\meshes_zopt_flow_prior_T085 --model VAE_zopt_a1_flow=%VAE%\meshes_zopt_a1_flow_prior ^
  --model real_mc=%CTRLV%\mc ^
  "data.manifests=['%FULL%']" data.path_mode=absolute data.splits=%SPLIT% ^
  data.bbox=data/processed/neuron-model/splits/bbox_splits_dev_100n.json ^
  data.drop_unassigned=true real.test_split=val "protocol.generation_seeds=[1]" real.morphometrics_table=%MORPH% name=generator-evaluation-anchored-val

echo ===== [L7/7] classifier ablation + components ===== %date% %time%
for /f "delims=" %%d in ('dir /b /ad /o-n runs\analysis\neuron-model\*_generator-evaluation-anchored-val') do (set EVAL=runs\analysis\neuron-model\%%d& goto :found)
:found
echo evaluation folder: %EVAL%
python scripts\neuron_model\classifier_ablation.py --eval-run %EVAL%
python scripts\neuron_model\classifier_ablation.py --eval-run %EVAL% --reference real_mc --models VAE_flow VAE_zopt_flow VAE_zopt_flow_T085 VAE_zopt_a1_flow
python scripts\neuron_model\diagnose_generated_components.py --eval-run %EVAL% --model VAE_flow=%VAE%\meshes_flow_prior ^
  --model VAE_zopt_flow=%VAE%\meshes_zopt_flow_prior --model VAE_zopt_flow_T085=%VAE%\meshes_zopt_flow_prior_T085 ^
  --model VAE_zopt_a1_flow=%VAE%\meshes_zopt_a1_flow_prior --model real_mc=%CTRLV%\mc

:end
echo ===== L done ===== %date% %time%
