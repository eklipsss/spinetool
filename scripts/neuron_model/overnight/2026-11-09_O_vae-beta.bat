@echo off
rem Overnight 2026-11-09, chain O (GPU ~12 h): (1) diagnostic of the decoder field, (2) VAE with a LARGER KL weight.
rem Evidence (chains J-M): the decoder draws a neck for the code fitted to a spine, a flow reproduces the distribution of such codes
rem   (holes 1.02-1.06 at the floor 1.00-1.03), yet meshes from prior samples have no necks; joint decoder + code fine-tuning (M)
rem   did not help. The latent is almost an auto-encoder (beta 1e-4: effective rank 13.5 of 128, KL of the fitted Gaussian to N(0, I)
rem   284 nats): the decoder is defined on a thin manifold. Only SMALLER beta (1e-5) was tried (J), a larger one was not.
rem O0 diagnostic (no training, ~30 min): perturbations of fitted codes by s*sigma, interpolation, flow samples pulled to the nearest
rem   fitted code -> is the field not smooth in the code (decoder), or do samples fall between codes (prior / structure)?
rem O1-O3 recipe of 2026-10-19_003408_vae-dev-reg0-cos (same config, seed, cache, split, 30 000 steps, cosine lr) with ONE factor changed:
rem   loss.beta 1e-4 -> 3e-4, 1e-3, 3e-3  (the baseline.yaml comment: beta 1e-3 can already let KL dominate -> watch active_latent_dimensions
rem   in logs\metrics_train.jsonl; a collapsed run is a result, not a failure). After each training: encoder codes of 15 000 train
rem   spines, flow prior, 200 meshes from the flow AND from N(0, I) (a better-behaved latent should need no learned prior).
rem O4 evaluation on the dev VAL neurons (VAE_flow, each beta with both priors, real_mc), classifier ablation, components,
rem   reconstruction bias of the new decoders (compare_vae_decoders, 300 val spines). The untouched test is NOT used.
rem Decision criterion (fixed in advance): for at least one beta, with the flow or the N(0, I) prior, single-component share >= 0.835 AND
rem   intact-mesh ConvexHullRatio clearly above -0.69 (real_mc -0.29) AND MMD of morphometrics <= 0.040. Otherwise the VAE is closed as a
rem   documented negative result and the effort goes to MoGen.
rem Log: scripts\neuron_model\overnight\2026-11-09_O_vae-beta.bat > logs_O.txt 2>&1

set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set VAE=runs\training\neuron-model\2026-10-19_003408_vae-dev-reg0-cos
set RUNS=runs\training\neuron-model
set SPLIT=data\processed\neuron-model\splits\splits_dev_100n_untouched_test.parquet
set CTRLV=runs\analysis\neuron-model\2026-11-08_real-controls-val
set FULL=C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/manifest_minnie65_full.parquet
set MORPH=C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/morphometrics_minnie65_full.parquet
set COMMON=train.max_steps=30000 train.validate_every=100000 train.checkpoint_every=2000 data.batch_size=16 loss.eikonal=0.1 loss.normal=0.3 optimizer.lr=1.0e-4 "optimizer.schedule={type: cosine, min_lr: 1.0e-5}"

cd /d %REPO%
if not exist %VAE%\config.yaml (echo MISSING %VAE%\config.yaml & goto :end)

echo ===== [O0] diagnostic of the decoder field around fitted codes ===== %date% %time%
python scripts\neuron_model\diagnose_decoder_smoothness.py --run %VAE% --codes-dir latent_prior_zopt_a1 --n 150
type %VAE%\smoothness_latent_prior_zopt_a1\summary.md

call :one_beta 3.0e-4 B1
if errorlevel 1 goto :end
call :one_beta 1.0e-3 B2
if errorlevel 1 goto :end
call :one_beta 3.0e-3 B3
if errorlevel 1 goto :end
echo runs: %RUN_B1% %RUN_B2% %RUN_B3%

echo ===== [O4] evaluation on the dev VAL neurons ===== %date% %time%
python scripts\neuron_model\evaluate_generators.py ^
  --model VAE_flow=%VAE%\meshes_flow_prior ^
  --model B3e-4_flow=%RUN_B1%\meshes_flow_prior --model B3e-4_normal=%RUN_B1%\meshes_normal_prior ^
  --model B1e-3_flow=%RUN_B2%\meshes_flow_prior --model B1e-3_normal=%RUN_B2%\meshes_normal_prior ^
  --model B3e-3_flow=%RUN_B3%\meshes_flow_prior --model B3e-3_normal=%RUN_B3%\meshes_normal_prior ^
  --model real_mc=%CTRLV%\mc ^
  "data.manifests=['%FULL%']" data.path_mode=absolute data.splits=%SPLIT% ^
  data.bbox=data/processed/neuron-model/splits/bbox_splits_dev_100n.json ^
  data.drop_unassigned=true real.test_split=val "protocol.generation_seeds=[1]" real.morphometrics_table=%MORPH% name=generator-evaluation-beta-val

for /f "delims=" %%d in ('dir /b /ad /o-n runs\analysis\neuron-model\*_generator-evaluation-beta-val') do (set EVAL=runs\analysis\neuron-model\%%d& goto :found)
:found
echo evaluation folder: %EVAL%
python scripts\neuron_model\classifier_ablation.py --eval-run %EVAL%
python scripts\neuron_model\classifier_ablation.py --eval-run %EVAL% --reference real_mc --models VAE_flow B3e-4_flow B3e-4_normal B1e-3_flow B1e-3_normal B3e-3_flow B3e-3_normal
python scripts\neuron_model\diagnose_generated_components.py --eval-run %EVAL% --model VAE_flow=%VAE%\meshes_flow_prior ^
  --model B3e-4_flow=%RUN_B1%\meshes_flow_prior --model B3e-4_normal=%RUN_B1%\meshes_normal_prior ^
  --model B1e-3_flow=%RUN_B2%\meshes_flow_prior --model B1e-3_normal=%RUN_B2%\meshes_normal_prior ^
  --model B3e-3_flow=%RUN_B3%\meshes_flow_prior --model B3e-3_normal=%RUN_B3%\meshes_normal_prior --model real_mc=%CTRLV%\mc
type %EVAL%\postprocessed\summary.md

echo ===== [O5] reconstruction bias of the new decoders, 300 dev-val spines ===== %date% %time%
python scripts\neuron_model\compare_vae_decoders.py --run base=%VAE% --run B3e-4=%RUN_B1% --run B1e-3=%RUN_B2% --run B3e-3=%RUN_B3% ^
  --mesh-manifest %FULL% --split val --n-spines 300 --resolution 128

goto :end

:one_beta
echo ===== [%2] beta %1: training ===== %date% %time%
python scripts\neuron_model\train_vae.py --config %VAE%\config.yaml name=vae-dev-beta%1 %COMMON% loss.beta=%1
if errorlevel 1 exit /b 1
for /f "delims=" %%d in ('dir /b /ad /o-d %RUNS%\*_vae-dev-beta%1') do if not defined RUN_%2 set RUN_%2=%RUNS%\%%d
call set CUR=%%RUN_%2%%
echo run: %CUR%
python scripts\neuron_model\analyze_vae_latent.py --run %CUR% --split val --n-spines 1500 --decode-spines 20 --n-posterior 8 --mesh-manifest %FULL%
python scripts\neuron_model\fit_latent_prior.py --run %CUR% --n-train 15000 --n-val 1500 --gmm 4 8
python scripts\neuron_model\generate_vae.py --run %CUR% --prior learned --latent-prior %CUR%\latent_prior\flow_prior.pt --n-samples 200 --seed 1 --resolution 128
python scripts\neuron_model\generate_vae.py --run %CUR% --prior normal --n-samples 200 --seed 1 --resolution 128 --out-name meshes_normal_prior
exit /b 0

:end
echo ===== O done ===== %date% %time%
