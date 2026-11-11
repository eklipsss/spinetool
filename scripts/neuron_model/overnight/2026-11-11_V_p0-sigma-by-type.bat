@echo off
rem 2026-11-11, chain V (CPU only, ~1-1.5 h): own spread of the neuron effect for inhibitory neurons in P0.
rem Reason (calibrate_p0_counts.py on the P0 of chain T, 2026-11-10_113713_p0, logs_calib.txt): on the TRAINING neurons the
rem   model matches the inhibitory ones (obs / model 1.15, empty 0.394 vs 0.392), on the 8 val inhibitory neurons 0.44;
rem   the neuron effect of inhibitory neurons spreads ~0.67 against the single sigma_b 0.285 (excitatory 0.30), so the model
rem   generates inhibitory neurons that all look alike. CountModel(sigma_groups=2): own sigma_b for inhibitory neurons,
rem   in the P0 grid (model.sigma_by_cell_type [false, true]), chosen by the val NLL as before.
rem V1 rarity of the 8 val inhibitory neurons under the OLD P0 (random 8-neuron subsets of the 47 training ones),
rem V2 P0 with the new grid, V3 its calibration, V4 P1 on it, V5 generation (T0 of chain T) + evaluation on val,
rem V6 real trunk on TRAIN branches, inhibitory evaluation (in-sample: does the generated spread match the training one).
rem Log: scripts\neuron_model\overnight\2026-11-11_V_p0-sigma-by-type.bat > logs_V.txt 2>&1

set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set B=C:\Users\Student\LipsEkaterina\data\minnie65\preprocessed_branches
set SPLIT=data\processed\neuron-model\splits\splits_v1.parquet
set RUNS=runs\training\neuron-model
set OLDP0=%RUNS%\2026-11-10_113713_p0
set T0=%RUNS%\2026-11-10_113612_t0
set TP=runs\analysis\neuron-model\tp_generation\sigma_by_type

cd /d %REPO%
if not exist %OLDP0%\p0_bundle.json (echo MISSING %OLDP0%\p0_bundle.json & goto :end)
if not exist %T0%\report.json (echo MISSING %T0%\report.json & goto :end)

rem V0 (minutes): the evaluation now thins the generated spines to the positioned share of their source branch for the
rem    per-spine statistics (as the real ones; before: ~34 nm of the nearest-neighbour gap was this measurement bias).
rem    Re-evaluate the sets of chains T and U into NEW folders evaluation_thinned* (the old evaluations stay).
rem    Expected: U1 (P1b2, real N) close to U2 of chain U (-22 nm); T6 generated / real_t: the count-model part only.
set TT=runs\analysis\neuron-model\tp_generation
echo ===== [V0] re-evaluation of the T6 and U1 sets with thinning ===== %date% %time%
for %%s in (%TT%\inhibitory\generated\2026-11-10_114858_generated_seed1 %TT%\inhibitory\real_t\2026-11-10_115009_real_t_seed1 %TT%\counts_real\p1b2\2026-11-10_174333_real_t_counts-real_seed1) do (
  echo ----- %%s -----
  python scripts\neuron_model\evaluate_assembled_branches.py --tp-dir %%s --out %%s\evaluation_thinned
  python scripts\neuron_model\evaluate_assembled_branches.py --tp-dir %%s --cell-type-binary excitatory --out %%s\evaluation_thinned_excitatory
  python scripts\neuron_model\evaluate_assembled_branches.py --tp-dir %%s --cell-type-binary inhibitory --out %%s\evaluation_thinned_inhibitory
)

echo ===== [V1] old P0: calibration + rarity of the val groups ===== %date% %time%
python scripts\neuron_model\calibrate_p0_counts.py --p0-run %OLDP0%

echo ===== [V2] P0 with own sigma_b for inhibitory neurons in the grid ===== %date% %time%
python scripts\neuron_model\fit_p0.py --config configs\neuron-model\placement\p0.yaml data.branches_root=%B% data.splits=%SPLIT%
if errorlevel 1 goto :end
for /f "delims=" %%d in ('dir /b /ad /o-n %RUNS%\*_p0') do (set P0=%RUNS%\%%d& goto :p0found)
:p0found
echo P0 run: %P0%

echo ===== [V3] calibration of the new P0 ===== %date% %time%
python scripts\neuron_model\calibrate_p0_counts.py --p0-run %P0%

echo ===== [V4] P1 on the new P0 ===== %date% %time%
python scripts\neuron_model\fit_p1.py --p0-run %P0% --n-eval 400 --n-sim 19
if errorlevel 1 goto :end
for /f "delims=" %%d in ('dir /b /ad /o-n %RUNS%\*_p1') do (set P1=%RUNS%\%%d& goto :p1found)
:p1found
echo P1 run: %P1%

echo ===== [V5] generated (T0 + P1) and real trunk on val, evaluation ===== %date% %time%
python scripts\neuron_model\generate_tp.py --t0-run %T0% --p0-run %P1% --mode generated --n 1000 --seed 1 --split val --out-root %TP%\generated
python scripts\neuron_model\generate_tp.py --t0-run %T0% --p0-run %P1% --mode real_t --n 1000 --seed 1 --split val --out-root %TP%\real_t
echo ===== [V6] real trunk on TRAIN branches (in-sample check of the inhibitory spread) ===== %date% %time%
python scripts\neuron_model\generate_tp.py --p0-run %P1% --mode real_t --n 3000 --seed 1 --split train --out-root %TP%\real_t_train
for %%r in (generated real_t) do (
  for /f "delims=" %%d in ('dir /b /ad %TP%\%%r') do (
    echo ----- %TP%\%%r\%%d: all / excitatory / inhibitory -----
    python scripts\neuron_model\evaluate_assembled_branches.py --tp-dir %TP%\%%r\%%d
    python scripts\neuron_model\evaluate_assembled_branches.py --tp-dir %TP%\%%r\%%d --cell-type-binary excitatory
    python scripts\neuron_model\evaluate_assembled_branches.py --tp-dir %TP%\%%r\%%d --cell-type-binary inhibitory
  )
)
for /f "delims=" %%d in ('dir /b /ad %TP%\real_t_train') do (
  echo ----- %TP%\real_t_train\%%d: inhibitory -----
  python scripts\neuron_model\evaluate_assembled_branches.py --tp-dir %TP%\real_t_train\%%d --cell-type-binary inhibitory
)

:end
echo ===== V done ===== %date% %time%
