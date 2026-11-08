@echo off
rem Overnight 2026-11-09, chain M (GPU ~6-8 h): the VAE decoder fine-tuned JOINTLY with per-spine codes (auto-decoder,
rem DeepSDF style), selected on the dev VAL neurons. Evidence (chain L, 2026-11-08_180137_generator-evaluation-anchored-val):
rem   codes fitted with the decoder FROZEN reconstruct their spines with necks, but a flow prior on them does not generate
rem   necks: intact meshes as convex as with encoder codes (ConvexHullRatio -0.69...-0.86 vs -0.69, real_mc -0.29), 27-42 %
rem   of the meshes break at the neck. The field AROUND a fitted code fills or breaks the neck -> the decoder must learn necks
rem   on the codes the prior is fitted to.
rem Two variants, everything else equal (start: base decoder + anchored codes w = 1 nm of chain L; 20 000 steps, batch 16,
rem decoder lr 3e-5 -> 3e-6 cosine, codes SparseAdam 5e-3, anchor 1 nm, the base run's SDF / far / eikonal / normal loss):
rem   n0  code noise 0          (plain DeepSDF: the decoder sees exactly the codes)
rem   n1  code noise 1 sigma    (z ~ N(c, sigma^2): the neighbourhood the VAE decoder was trained on, around a free code)
rem Then for each: validation codes with the fine-tuned decoder frozen (as the train codes started: from mu, w = 1), flow,
rem 200 meshes; evaluation on the dev VAL neurons with VAE_flow, VAE_zopt_a1_flow and real_mc (same as chain L); finally the
rem reconstruction bias of fitted codes (compare_vae_decoders, 300 dev-val spines). The untouched test is NOT used here.
rem Decision criterion: single-component share >= that of VAE_flow (0.835) AND intact-mesh ConvexHullRatio clearly above
rem -0.69 (toward real_mc -0.29) AND MMD morphometrics not worse than VAE_flow (0.040).
rem Log: scripts\neuron_model\overnight\2026-11-09_M_vae-autodecoder.bat > logs_M.txt 2>&1

set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set VAE=runs\training\neuron-model\2026-10-19_003408_vae-dev-reg0-cos
set RUNS=runs\training\neuron-model
set SPLIT=data\processed\neuron-model\splits\splits_dev_100n_untouched_test.parquet
set CTRLV=runs\analysis\neuron-model\2026-11-08_real-controls-val
set FULL=C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/manifest_minnie65_full.parquet
set MORPH=C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/morphometrics_minnie65_full.parquet

cd /d %REPO%

echo ===== [M1/9] fine-tune decoder + codes, code noise 0 ===== %date% %time%
python scripts\neuron_model\finetune_vae_autodecoder.py --base-run %VAE% --codes-dir latent_prior_zopt_a1 --name vae-autodecoder-n0 ^
  --steps 20000 --batch-size 16 --code-noise 0 --anchor-weight 1 --num-workers 8
if errorlevel 1 goto :end
for /f "delims=" %%d in ('dir /b /ad /o-n %RUNS%\*_vae-autodecoder-n0') do (set AD0=%RUNS%\%%d& goto :found0)
:found0
echo AD0 = %AD0%

echo ===== [M2/9] fine-tune decoder + codes, code noise 1 sigma ===== %date% %time%
python scripts\neuron_model\finetune_vae_autodecoder.py --base-run %VAE% --codes-dir latent_prior_zopt_a1 --name vae-autodecoder-n1 ^
  --steps 20000 --batch-size 16 --code-noise 1 --anchor-weight 1 --num-workers 8
if errorlevel 1 goto :end
for /f "delims=" %%d in ('dir /b /ad /o-n %RUNS%\*_vae-autodecoder-n1') do (set AD1=%RUNS%\%%d& goto :found1)
:found1
echo AD1 = %AD1%

echo ===== [M3/9] validation codes, fine-tuned decoders frozen ===== %date% %time%
python scripts\neuron_model\fit_vae_codes.py --run %AD0% --splits-to-fit val --n-val 1500 --steps 500 --batch-size 32 --anchor-weight 1 --out-name latent_prior_ad
if errorlevel 1 goto :end
python scripts\neuron_model\fit_vae_codes.py --run %AD1% --splits-to-fit val --n-val 1500 --steps 500 --batch-size 32 --anchor-weight 1 --out-name latent_prior_ad
if errorlevel 1 goto :end

echo ===== [M4/9] flow priors ===== %date% %time%
python scripts\neuron_model\fit_latent_prior.py --run %AD0% --variants 1 --out-name latent_prior_ad --n-train 15000 --n-val 1500 --gmm 4 8
if errorlevel 1 goto :end
python scripts\neuron_model\fit_latent_prior.py --run %AD1% --variants 1 --out-name latent_prior_ad --n-train 15000 --n-val 1500 --gmm 4 8
if errorlevel 1 goto :end

echo ===== [M5/9] 200 meshes from each flow, MC 128^3 ===== %date% %time%
python scripts\neuron_model\generate_vae.py --run %AD0% --prior learned --latent-prior %AD0%\latent_prior_ad\flow_prior.pt ^
  --n-samples 200 --seed 1 --resolution 128 --out-name meshes_ad_flow_prior
python scripts\neuron_model\generate_vae.py --run %AD1% --prior learned --latent-prior %AD1%\latent_prior_ad\flow_prior.pt ^
  --n-samples 200 --seed 1 --resolution 128 --out-name meshes_ad_flow_prior

echo ===== [M6/9] evaluation on the dev VAL neurons ===== %date% %time%
python scripts\neuron_model\evaluate_generators.py ^
  --model VAE_flow=%VAE%\meshes_flow_prior --model VAE_zopt_a1_flow=%VAE%\meshes_zopt_a1_flow_prior ^
  --model AD_n0_flow=%AD0%\meshes_ad_flow_prior --model AD_n1_flow=%AD1%\meshes_ad_flow_prior ^
  --model real_mc=%CTRLV%\mc ^
  "data.manifests=['%FULL%']" data.path_mode=absolute data.splits=%SPLIT% ^
  data.bbox=data/processed/neuron-model/splits/bbox_splits_dev_100n.json ^
  data.drop_unassigned=true real.test_split=val "protocol.generation_seeds=[1]" real.morphometrics_table=%MORPH% name=generator-evaluation-autodecoder-val

echo ===== [M7/9] classifier ablation ===== %date% %time%
for /f "delims=" %%d in ('dir /b /ad /o-n runs\analysis\neuron-model\*_generator-evaluation-autodecoder-val') do (set EVAL=runs\analysis\neuron-model\%%d& goto :found)
:found
echo evaluation folder: %EVAL%
python scripts\neuron_model\classifier_ablation.py --eval-run %EVAL%
python scripts\neuron_model\classifier_ablation.py --eval-run %EVAL% --reference real_mc --models VAE_flow VAE_zopt_a1_flow AD_n0_flow AD_n1_flow

echo ===== [M8/9] components ===== %date% %time%
python scripts\neuron_model\diagnose_generated_components.py --eval-run %EVAL% --model VAE_flow=%VAE%\meshes_flow_prior ^
  --model VAE_zopt_a1_flow=%VAE%\meshes_zopt_a1_flow_prior --model AD_n0_flow=%AD0%\meshes_ad_flow_prior ^
  --model AD_n1_flow=%AD1%\meshes_ad_flow_prior --model real_mc=%CTRLV%\mc
type %EVAL%\postprocessed\summary.md

echo ===== [M9/9] reconstruction bias of fitted codes, 300 dev-val spines ===== %date% %time%
python scripts\neuron_model\compare_vae_decoders.py --run base=%VAE% --run ad_n0=%AD0% --run ad_n1=%AD1% ^
  --optimize-latent 500 --opt-runs base ad_n0 ad_n1 --opt-anchor-weights 1 ^
  --mesh-manifest %FULL% --split val --n-spines 300 --resolution 128

:end
echo ===== M done ===== %date% %time%
