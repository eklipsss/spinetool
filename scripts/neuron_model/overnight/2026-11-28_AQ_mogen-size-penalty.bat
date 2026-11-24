@echo off
setlocal enabledelayedexpansion
rem 2026-11-28, chain AQ. Size penalty in the MoGen fine-tuning + model A on the larger training part. GPU (after AP),
rem roughly 1-1.5 days in total (three trainings; with the penalty a step costs about twice); CPU for the meshes and the
rem evaluation at the end. Unconditional, as A: a clean test of one change; a conditional model would then start from
rem the corrected A (as v1-v4 started from A) with the same penalty in its config.
rem
rem Why: the generated clouds are 10-12 %% too thick from the spine axis with the right length; a plateau of the
rem fine-tuning (AJ), created by the field itself (AM: 50-400 ODE steps give the same size; a smaller starting noise
rem destroys the cloud). Penalty (flow.size_penalty_loss): one cloud per micro-batch integrated from the TRUE x_t
rem (t0 ~ U(0, 0.8)) to t = 1 in 4 midpoint steps WITH gradients; the squared gap of the batch means of log radial and
rem log axial size against the real clouds of the batch, weight 3. Validation logs val_size_gap_radial / _axial.
rem AQ1 A continued from its latest weights with the penalty, 6 000 steps, the same 15 000 spines (dev cache)
rem AQ2 A from mouse_mixed on 60 000 training spines of splits_v1 (cache v1_60k) WITH the penalty, 12 000 steps
rem AQ3 the same WITHOUT the penalty (control: does more data alone change the size?)
rem     If a training runs out of GPU memory: flow.size_penalty.n_steps=2 (or data.batch_size=4 train.accumulation_steps=8).
rem AQ4 300 clouds per model (no meshes) -> cloud_size_diagnostics against the real clouds and A
rem AQ5 500 x 2 seeds per model -> Poisson d8_pw8 (separate process) -> evaluate_generators as chain AI + quality report
rem Log: scripts\neuron_model\overnight\2026-11-28_AQ_mogen-size-penalty.bat > logs_AQ.txt 2>&1

set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set SPLITS=data\processed\neuron-model\splits
set RUNS=runs\training\neuron-model
set ANA=runs\analysis\neuron-model
set MA=%RUNS%\2026-10-20_213103_mogen-ft-dev
set MAN=%MA%\meshes_poisson_d8_pw8_n500
set CMP2=%ANA%\2026-11-21_v3-vs-A-seeds
set CACHE=cache/minnie65_v1_60k/manifest.parquet
set CFGP=configs\neuron-model\reconstruction\poisson_sweep\d8_pw8.yaml
set FULL=C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/manifest_minnie65_full.parquet
set MORPH=C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/morphometrics_minnie65_full.parquet
set PEN=flow.size_penalty={weight: 3.0, every: 1, n_sub: 1, t_min: 0.0, t_max: 0.8, n_steps: 4, weights: [1.0, 1.0]}
set MON=flow.size_monitor={n_batches: 10, t_min: 0.0, t_max: 0.8, n_steps: 10}

cd /d %REPO%
if not exist %MA%\config.yaml (echo NO run A & goto :end)

echo ===== [AQ1] A continued with the size penalty ===== %date% %time%
python scripts\neuron_model\train_mogen.py --config %MA%\config.yaml name=mogen-a-cont-size model.init=run model.init_run=%MA% ^
  model.init_checkpoint=latest train.max_steps=6000 train.validate_every=2000 train.checkpoint_every=2000 "%PEN%" "%MON%"
echo ===== [AQ2] A on splits_v1 (60 000) with the size penalty ===== %date% %time%
python scripts\neuron_model\train_mogen.py --config configs\neuron-model\mogen\a_full_size.yaml "data.manifests=['%CACHE%']" data.splits=%SPLITS%\splits_v1.parquet
echo ===== [AQ3] A on splits_v1 (60 000) without the penalty ===== %date% %time%
python scripts\neuron_model\train_mogen.py --config configs\neuron-model\mogen\a_full.yaml "data.manifests=['%CACHE%']" data.splits=%SPLITS%\splits_v1.parquet

set GEN=
for %%n in (a-cont-size a-full-size a-full) do (
  for /f "delims=" %%d in ('dir /b /ad /o-n %RUNS%\*_mogen-%%n 2^>nul') do (
    if not defined RUN_%%n set RUN_%%n=%RUNS%\%%d
  )
)
echo runs: !RUN_a-cont-size! / !RUN_a-full-size! / !RUN_a-full!

echo ===== [AQ4] 300 clouds per model and their size ===== %date% %time%
set SIZEGEN=
for %%n in (a-cont-size a-full-size a-full) do (
  if defined RUN_%%n (
    python scripts\neuron_model\generate_mogen.py --run !RUN_%%n! --checkpoint latest --n-samples 300 --seed 1 --tag size_seed_1 --no-mesh
    set SIZEGEN=!SIZEGEN! --gen %%n=!RUN_%%n!\meshes\size_seed_1
  )
)
python scripts\neuron_model\cloud_size_diagnostics.py --manifest cache\minnie65_v1_60k\manifest.parquet --real real=%CMP2%\real\poisson\seed_1 ^
  --gen A=%MAN%\seed_1 !SIZEGEN! --out %ANA%\2026-11-28_size-penalty-clouds

echo ===== [AQ5] meshes and evaluation ===== %date% %time%
set MODELS=--model mogen_A=%MAN% --model real_poisson=%CMP2%\real\poisson
for %%n in (a-cont-size a-full-size a-full) do (
  if defined RUN_%%n (
    for %%s in (1 2) do (
      python scripts\neuron_model\generate_mogen.py --run !RUN_%%n! --checkpoint latest --n-samples 500 --seed %%s --no-mesh
      python scripts\neuron_model\reconstruct_mogen_poisson.py --run !RUN_%%n! --seed %%s --meshes-dir !RUN_%%n!\meshes --out !RUN_%%n!\meshes_poisson_d8_pw8 ^
        --poisson-config %CFGP% --workers 8
    )
    set MODELS=!MODELS! --model %%n=!RUN_%%n!\meshes_poisson_d8_pw8
  )
)
python scripts\neuron_model\evaluate_generators.py !MODELS! ^
  "data.manifests=['%FULL%']" data.path_mode=absolute data.splits=%SPLITS%\splits_v1_val_minus_dev.parquet ^
  data.bbox=%SPLITS%\bbox_splits_v1.json data.drop_unassigned=true "protocol.generation_seeds=[1,2]" real.morphometrics_table=%MORPH% ^
  real.chamfer_memorization=false name=generator-evaluation-size-penalty
for /f "delims=" %%d in ('dir /b /ad /o-n %ANA%\*_generator-evaluation-size-penalty') do (set EVAL=%ANA%\%%d& goto :evalfound)
:evalfound
echo evaluation run: %EVAL%
python scripts\neuron_model\make_quality_report.py --out %EVAL%\quality --s-eval %EVAL%
type %EVAL%\postprocessed\summary.md

:end
echo ===== chain AQ done ===== %date% %time%
