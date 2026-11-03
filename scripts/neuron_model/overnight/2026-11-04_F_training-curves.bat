@echo off
rem Overnight 2026-11-04, chain F (GPU): TRAIN / VALIDATION CURVES with one estimator (teacher's request: loss curves with uncertainty).
rem Run from any folder, env neuron-model active. ~5.5-6 h in total.
rem   1. VAE: RE-RUN of the recipe of 2026-10-19_003408_vae-dev-reg0-cos (30 000 steps, cosine lr 1e-4 -> 1e-5, eikonal 0.1 / normal 0.3
rem      from step 0, same seed, same cache manifest and splits_dev_100n) with a snapshot every 1000 steps and NO validation in the loop
rem      (validate_every=100000: in-loop validation slows the Windows training ~10x for good)            ~2.5 h
rem   2. eval_training_curves.py on that run: every snapshot, EMA/raw weights (the VAE has no EMA -> raw), fixed val spines and a fixed
rem      train subset of the same size, all loss components, KL weight at the step and at the final step                      ~15-20 min
rem   3. MoGen A: same for the fine-tune of mouse_mixed (16 000 steps, config of 2026-10-20_213103_mogen-ft-dev), in-loop validation
rem      kept (every 2000 steps), snapshot every 1000 steps                                                                    ~2.6 h (1.73 steps/s)
rem   4. eval_training_curves.py on it (EMA weights; 3 fixed (t, noise) draws)                                                  ~15-25 min
rem Post-hoc evaluation time per snapshot is NOT measured on the real models (the mac only runs the tiny test model, ~1 s for 8 spines):
rem extrapolated from the training speed (forward only, ~3x faster than a training step): 1000 + 1000 spines per snapshot ~ 30-60 s,
rem 30 (VAE) / 16 (A) snapshots. If it is slower, --n-spines 500 halves it; an interrupted evaluation continues from curves.json.
rem Disk: one snapshot = model + EMA + optimizer state. VAE: small (~0.1-0.2 GB x 30). MoGen A: ~4 x 224 MB ~ 0.9 GB x 16 = ~15 GB. Check free space on C:.
rem
rem THE ORIGINAL RUNS STAY UNTOUCHED. The re-run is a repetition of the same recipe in a NEW run folder (name ...-curves); everything obtained
rem with the old runs (meshes, atlas, latent prior, branch assembly) keeps referring to them. Training is not bit-reproducible on GPU:
rem the repeated run is the same recipe, not the same weights; curves describe THIS run.
rem The recipe is taken from the saved config.yaml of the original run (data paths, seed, model, splits, bbox, variants...) and the keys that
rem define the recipe are repeated explicitly below. The saved config is printed to the log - compare it with the registry before trusting it
rem (a resumed original run may have had overrides on the command line that config.yaml does not contain).
rem Log: scripts\neuron_model\overnight\2026-11-04_F_training-curves.bat > logs_F.txt 2>&1

set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set OLD_VAE=runs\training\neuron-model\2026-10-19_003408_vae-dev-reg0-cos
set OLD_A=runs\training\neuron-model\2026-10-20_213103_mogen-ft-dev

cd /d %REPO%

echo ===== [F0] original configs ===== %date% %time%
if not exist %OLD_VAE%\config.yaml (echo MISSING %OLD_VAE%\config.yaml & goto :end)
if not exist %OLD_A%\config.yaml (echo MISSING %OLD_A%\config.yaml & goto :end)
type %OLD_VAE%\config.yaml
type %OLD_A%\config.yaml
echo metrics_val.jsonl of the original runs (in-loop validation): 
dir %OLD_A%\logs\metrics_val.jsonl %OLD_VAE%\logs\metrics_val.jsonl

echo ===== [F1/4] VAE re-run with snapshots ===== %date% %time%
python scripts\neuron_model\train_vae.py --config %OLD_VAE%\config.yaml name=vae-dev-reg0-cos-curves ^
  train.max_steps=30000 train.validate_every=100000 train.keep_every=1000 train.checkpoint_every=1000 ^
  data.batch_size=16 loss.eikonal=0.1 loss.normal=0.3 optimizer.lr=1.0e-4 "optimizer.schedule={type: cosine, min_lr: 1.0e-5}"

rem the newest curves run folder
for /f "delims=" %%d in ('dir /b /ad /o-d runs\training\neuron-model\*_vae-dev-reg0-cos-curves') do (set VAE=runs\training\neuron-model\%%d& goto :vfound)
:vfound
echo VAE curves run: %VAE%

echo ===== [F2/4] VAE curves ===== %date% %time%
python scripts\neuron_model\eval_training_curves.py --run %VAE% --weights ema --n-spines 1000

echo ===== [F3/4] MoGen A re-run with snapshots ===== %date% %time%
python scripts\neuron_model\train_mogen.py --config %OLD_A%\config.yaml name=mogen-ft-dev-curves ^
  train.max_steps=16000 train.validate_every=100000 train.keep_every=1000 train.checkpoint_every=1000

for /f "delims=" %%d in ('dir /b /ad /o-d runs\training\neuron-model\*_mogen-ft-dev-curves') do (set A=runs\training\neuron-model\%%d& goto :afound)
:afound
echo MoGen A curves run: %A%

echo ===== [F4/4] MoGen A curves ===== %date% %time%
python scripts\neuron_model\eval_training_curves.py --run %A% --weights ema --n-spines 1000 --n-draws 3

echo Open: %VAE%\training_curves\curves.html and %A%\training_curves\curves.html
echo Record both run folders in docs\neuron-model\runs-and-outputs-registry.md
:end
echo ===== F done ===== %date% %time%
