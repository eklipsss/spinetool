@echo off
rem 2026-11-10, chain U (CPU only, ~10 min): why is the surface nearest neighbour of generated spines ~75-100 nm shorter
rem   than the real one (chain T, T6) although P1b2 matches it at the REAL count (fit_p1 Monte Carlo check: 963 vs 983 nm)?
rem   The only difference between the two checks is N per branch: the P1 check uses the real N of each branch, generate_tp
rem   samples N from the ZINB count model. Here the real trunk (real_t, same seed and branches as T6) gets
rem   U1 N = n_spines_in_range (the real count the evaluation compares with),
rem   U2 N = n_with_position (the spines with a position: the ones the real distances are computed on = the P1 check),
rem   U3 as U2 but with the P0 positions (no interaction): how much of the gap P1b2 itself closes.
rem   Reading: gap gone in U2 but not in U1 -> the real distances are biased long (~7 %% of spines have no position);
rem            gap gone in U1 and U2 -> the count model (N per branch) makes it; gap stays in U2 -> the positions.
rem Log: scripts\neuron_model\overnight\2026-11-10_U_counts-real.bat > logs_U.txt 2>&1

set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set RUNS=runs\training\neuron-model
set P0=%RUNS%\2026-11-10_113713_p0
set P1=%RUNS%\2026-11-10_114732_p1
set OUT=runs\analysis\neuron-model\tp_generation\counts_real

cd /d %REPO%
if not exist %P1%\p0_bundle.json (echo MISSING %P1%\p0_bundle.json & goto :end)

echo ===== [U1] real trunk, P1b2, N = n_spines_in_range ===== %date% %time%
python scripts\neuron_model\generate_tp.py --p0-run %P1% --mode real_t --counts real --n 1000 --seed 1 --split val --out-root %OUT%\p1b2
echo ===== [U2] real trunk, P1b2, N = n_with_position ===== %date% %time%
python scripts\neuron_model\generate_tp.py --p0-run %P1% --mode real_t --counts real_positioned --n 1000 --seed 1 --split val --out-root %OUT%\p1b2
echo ===== [U3] real trunk, P0 positions, N = n_with_position ===== %date% %time%
python scripts\neuron_model\generate_tp.py --p0-run %P0% --mode real_t --counts real_positioned --n 1000 --seed 1 --split val --out-root %OUT%\p0

for /f "delims=" %%d in ('dir /b /ad /s %OUT%\*_seed1') do (
  echo ----- %%d: all / excitatory / inhibitory -----
  python scripts\neuron_model\evaluate_assembled_branches.py --tp-dir %%d
  python scripts\neuron_model\evaluate_assembled_branches.py --tp-dir %%d --cell-type-binary excitatory
  python scripts\neuron_model\evaluate_assembled_branches.py --tp-dir %%d --cell-type-binary inhibitory
)

:end
echo ===== U done ===== %date% %time%
