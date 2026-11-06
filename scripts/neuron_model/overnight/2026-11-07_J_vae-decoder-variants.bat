@echo off
rem Overnight 2026-11-07, chain J (GPU, ~9 h): can the VAE DECODER be made less biased?
rem Evidence (chain I, untouched test): against real spines through the same exact-SDF + MC chain the VAE is still told apart
rem with AUC 0.95 (0.98 at 256^3); reconstructions of REAL codes are already biased (ConvexHullRatio -0.97 SD = too convex,
rem necks filled; base area +0.88 SD). Neither the prior nor the MC grid is the cause. Three hypotheses, ONE factor per run,
rem everything else = the recipe of 2026-10-19_003408_vae-dev-reg0-cos (same config file, seed, cache, split, 30 000 steps, cosine lr):
rem   J1 beta 1e-4 -> 1e-5   posterior noise: the decoder is trained on z = mu + sigma*eps with sigma ~0.42 of the spread
rem                          between spines, so it learns the AVERAGE of neighbouring shapes (VAE blur fills concavities)
rem   J2 normal 0.3 -> 0.1   the normal-alignment regulariser smooths thin necks (eikonal 0.1 kept: without both the field
rem                          was rippled, see reg.yaml)
rem   J3 first_omega_0 30 -> 60   spectral bias of SIREN: features of ~100 nm (necks) need higher spatial frequencies
rem J4: compare_vae_decoders.py on 300 dev-VALIDATION spines (selection on val; the untouched test is used once, for the
rem winner, after its flow prior is refitted). The two trainings of the same recipe (original + the 2026-11-04 curves re-run)
rem give the training-noise floor: a variant must beat the baseline by more than the re-run differs from it.
rem Log: scripts\neuron_model\overnight\2026-11-07_J_vae-decoder-variants.bat > logs_J.txt 2>&1

set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set OLD_VAE=runs\training\neuron-model\2026-10-19_003408_vae-dev-reg0-cos
set RERUN=runs\training\neuron-model\2026-11-04_015937_vae-dev-reg0-cos-curves
set FULL=C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/manifest_minnie65_full.parquet
set COMMON=train.max_steps=30000 train.validate_every=100000 train.checkpoint_every=2000 data.batch_size=16 loss.eikonal=0.1 loss.normal=0.3 optimizer.lr=1.0e-4 "optimizer.schedule={type: cosine, min_lr: 1.0e-5}"

cd /d %REPO%
if not exist %OLD_VAE%\config.yaml (echo MISSING %OLD_VAE%\config.yaml & goto :end)

echo ===== [J1/4] beta 1e-5 ===== %date% %time%
python scripts\neuron_model\train_vae.py --config %OLD_VAE%\config.yaml name=vae-dev-beta1e-5 %COMMON% loss.beta=1.0e-5
echo ===== [J2/4] normal 0.1 ===== %date% %time%
python scripts\neuron_model\train_vae.py --config %OLD_VAE%\config.yaml name=vae-dev-normal0.1 %COMMON% loss.normal=0.1
echo ===== [J3/4] omega 60 ===== %date% %time%
python scripts\neuron_model\train_vae.py --config %OLD_VAE%\config.yaml name=vae-dev-omega60 %COMMON% model.decoder.first_omega_0=60

for /f "delims=" %%d in ('dir /b /ad /o-d runs\training\neuron-model\*_vae-dev-beta1e-5') do (set J1=runs\training\neuron-model\%%d& goto :f1)
:f1
for /f "delims=" %%d in ('dir /b /ad /o-d runs\training\neuron-model\*_vae-dev-normal0.1') do (set J2=runs\training\neuron-model\%%d& goto :f2)
:f2
for /f "delims=" %%d in ('dir /b /ad /o-d runs\training\neuron-model\*_vae-dev-omega60') do (set J3=runs\training\neuron-model\%%d& goto :f3)
:f3
echo runs: %J1% %J2% %J3%

echo ===== [J4/4] decoder study on 300 validation spines ===== %date% %time%
python scripts\neuron_model\compare_vae_decoders.py --run base=%OLD_VAE% --run base_rerun=%RERUN% ^
  --run beta1e-5=%J1% --run normal0.1=%J2% --run omega60=%J3% ^
  --mesh-manifest %FULL% --split val --n-spines 300 --resolution 128

:end
echo ===== J done ===== %date% %time%
