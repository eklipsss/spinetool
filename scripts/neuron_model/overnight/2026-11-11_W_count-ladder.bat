@echo off
rem 2026-11-11, chain W (CPU only, ~40-60 min): where does the ~40 nm of the nearest-neighbour gap that chain U attributed to
rem   the count model come from? Run AFTER chain V (uses its newest P1 run).
rem Ladder on the same 1000 val source branches (same seed = same requests), real trunk, positions from P1:
rem   model        N from the count model, fresh neuron effects (= V5 real_t)
rem   model_neuron N from the count model, effect of each source neuron drawn from its posterior given the observed
rem                counts of all its val branches (generate_tp.py --counts model_neuron)
rem   real         N = n_spines_in_range of the source branch
rem model -> model_neuron = neuron level (the 74 val neurons are poorer / richer than the population; calibration of V:
rem   val excitatory obs / model 0.95, an ordinary subset of training neurons), model_neuron -> real = branch level
rem   (covariate shape, dispersion of N inside a neuron). The evaluation thins the generated spines to the positioned share
rem   of their source branch (since 2026-11-11), so real vs real_positioned no longer differ by the measurement bias.
rem W1 also prints the branch-level calibration (observed vs simulated density, empty share and SPINE-WEIGHTED density by
rem   radius / length / distance to the soma / terminal / compartment / cell type, fresh vs posterior neuron effects).
rem W0 repeats the inhibitory evaluation of V6 (it ran out of memory in load_tp_set, fixed 2026-11-11).
rem Log: scripts\neuron_model\overnight\2026-11-11_W_count-ladder.bat > logs_W.txt 2>&1

set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set RUNS=runs\training\neuron-model
set TPV=runs\analysis\neuron-model\tp_generation\sigma_by_type
set TP=runs\analysis\neuron-model\tp_generation\counts_ladder

cd /d %REPO%
for /f "delims=" %%d in ('dir /b /ad /o-n %RUNS%\*_p1') do (set P1=%RUNS%\%%d& goto :p1found)
:p1found
if not exist %P1%\p0_bundle.json (echo MISSING %P1%\p0_bundle.json & goto :end)
echo P1 run: %P1%

echo ===== [W0] V6 again: real trunk on TRAIN branches, inhibitory ===== %date% %time%
for /f "delims=" %%d in ('dir /b /ad %TPV%\real_t_train') do (
  echo ----- %TPV%\real_t_train\%%d: inhibitory -----
  python scripts\neuron_model\evaluate_assembled_branches.py --tp-dir %TPV%\real_t_train\%%d --cell-type-binary inhibitory
)

echo ===== [W1] calibration of the count model with the branch level ===== %date% %time%
python scripts\neuron_model\calibrate_p0_counts.py --p0-run %P1% --n-sim-branch 100

echo ===== [W2] count ladder on val (real trunk, P1 positions) ===== %date% %time%
for %%c in (model model_neuron real) do (
  python scripts\neuron_model\generate_tp.py --p0-run %P1% --mode real_t --counts %%c --n 1000 --seed 1 --split val --out-root %TP%\%%c
)
for %%c in (model model_neuron real) do (
  for /f "delims=" %%d in ('dir /b /ad %TP%\%%c') do (
    echo ----- %%c: %TP%\%%c\%%d: all / excitatory / inhibitory -----
    python scripts\neuron_model\evaluate_assembled_branches.py --tp-dir %TP%\%%c\%%d
    python scripts\neuron_model\evaluate_assembled_branches.py --tp-dir %TP%\%%c\%%d --cell-type-binary excitatory
    python scripts\neuron_model\evaluate_assembled_branches.py --tp-dir %TP%\%%c\%%d --cell-type-binary inhibitory
  )
)

:end
echo ===== W done ===== %date% %time%
