@echo off
rem Overnight 2026-11-09, chain Q (CPU only, ~1-2 h; can run IN PARALLEL with chains O (GPU) and P (CPU)): P1 placement models.
rem Evidence: real spines repel each other at short range (g(0-0.5 um) = 0.76); the assembled branches with P0 place spines less
rem   regularly than real ones (nearest neighbour -94 nm, intervals -200 nm, significant with neurons as the unit), and with the REAL
rem   trunk and generated P the intervals are 1 188 vs 1 370 nm: the residual is in P, which places the N spines of a branch independently.
rem Q1 fit_p1: P1a (renewal along the branch, Weibull k) and P1b (Strauss on the trunk surface, R and gamma by pseudo-likelihood) on
rem   top of the current P0 (direction compartments); Monte Carlo goodness of fit on 400 validation branches (g(r), intervals, surface
rem   nearest neighbour) -> chosen model = p0_bundle.json of the P1 run.
rem Q2 real trunk + generated P with P0, P1a, P1b (same 400 requests, same seed): the P level alone.
rem Q3 evaluate_assembled_branches on each set (median differences with a bootstrap over NEURONS against the real branches).
rem Log: scripts\neuron_model\overnight\2026-11-09_Q_p1-placement.bat > logs_Q.txt 2>&1

set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set P0=runs\training\neuron-model\2026-11-03_203955_p0
set RUNS=runs\training\neuron-model
set TP=runs\analysis\neuron-model\tp_generation

cd /d %REPO%
if not exist %P0%\p0_bundle.json (echo MISSING %P0%\p0_bundle.json & goto :end)

echo ===== [Q1] fit P1a / P1b, simulation check ===== %date% %time%
python scripts\neuron_model\fit_p1.py --p0-run %P0% --n-eval 400 --n-sim 19
if errorlevel 1 goto :end
for /f "delims=" %%d in ('dir /b /ad /o-n %RUNS%\*_p1') do (set P1=%RUNS%\%%d& goto :found)
:found
echo P1 run: %P1%

echo ===== [Q2] real trunk + generated P: P0, P1a, P1b ===== %date% %time%
python scripts\neuron_model\generate_tp.py --p0-run %P0% --mode real_t --n 400 --seed 1 --split val --out-root %TP%\p1_compare\p0
python scripts\neuron_model\generate_tp.py --p0-run %P1% --bundle p1a_bundle.json --mode real_t --n 400 --seed 1 --split val --out-root %TP%\p1_compare\p1a
python scripts\neuron_model\generate_tp.py --p0-run %P1% --bundle p1b_bundle.json --mode real_t --n 400 --seed 1 --split val --out-root %TP%\p1_compare\p1b

echo ===== [Q3] evaluation against the real branches ===== %date% %time%
for %%m in (p0 p1a p1b) do (
  for /f "delims=" %%d in ('dir /b /ad /o-n %TP%\p1_compare\%%m') do (
    echo ----- %%m: %%d -----
    python scripts\neuron_model\evaluate_assembled_branches.py --tp-dir %TP%\p1_compare\%%m\%%d
  )
)

:end
echo ===== Q done ===== %date% %time%
