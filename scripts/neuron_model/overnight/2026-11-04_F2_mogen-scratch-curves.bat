@echo off
rem OPTIONAL, 2026-11-04, chain F2 (GPU): train / validation curves of MoGen B (from scratch, 60 000 steps, config of
rem 2026-10-21_001448_mogen-scratch-dev). The original run has NO validation curve (only the final point at step 60000), so this chain is REQUIRED for the curves. ~9.6 h of training + ~30-45 min of post-hoc evaluation (30 snapshots, EMA weights, 3 draws;
rem not measured on the real model, extrapolated from the training speed). May run in parallel with chain F (two windows): check nvidia-smi memory and the steps/s of both; 60 000 steps still finish in 18 h at 0.9 steps/s. In-loop validation is OFF (validate_every=100000): the post-hoc evaluation computes train and validation with ONE estimator.
rem Snapshot every 2000 steps (= the evaluation grid): 30 snapshots x ~0.9 GB (model + EMA + optimizer) ~ 27 GB - check free space on C:.
rem The ORIGINAL run 2026-10-21_001448_mogen-scratch-dev stays untouched; this is a repetition of its recipe in a new run folder.
rem Log: scripts\neuron_model\overnight\2026-11-04_F2_mogen-scratch-curves.bat > logs_F2.txt 2>&1

set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set OLD_B=runs\training\neuron-model\2026-10-21_001448_mogen-scratch-dev

cd /d %REPO%

if not exist %OLD_B%\config.yaml (echo MISSING %OLD_B%\config.yaml & goto :end)
type %OLD_B%\config.yaml
dir %OLD_B%\logs\metrics_val.jsonl

echo ===== [F2-1/2] MoGen B re-run with snapshots ===== %date% %time%
python scripts\neuron_model\train_mogen.py --config %OLD_B%\config.yaml name=mogen-scratch-dev-curves ^
  train.max_steps=60000 train.validate_every=100000 train.keep_every=2000 train.checkpoint_every=2000

for /f "delims=" %%d in ('dir /b /ad /o-d runs\training\neuron-model\*_mogen-scratch-dev-curves') do (set B=runs\training\neuron-model\%%d& goto :bfound)
:bfound
echo MoGen B curves run: %B%

echo ===== [F2-2/2] MoGen B curves ===== %date% %time%
python scripts\neuron_model\eval_training_curves.py --run %B% --weights ema --n-spines 1000 --n-draws 3

echo Open: %B%\training_curves\curves.html ; record the run folder in docs\neuron-model\runs-and-outputs-registry.md
:end
echo ===== F2 done ===== %date% %time%
