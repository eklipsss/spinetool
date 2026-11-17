@echo off
rem 2026-11-17, chain AC (CPU only, ~2-3 h). Run AFTER chain AB has finished, after git pull.
rem
rem T0 bending, the remaining defect of chain AA: the real kappa autocorrelation at 1 um rises with the radius (thin 0.16-0.29,
rem thick 0.52-0.60), the model's wiggle scale l_w was shared (+0.064 in all 5 CV folds). Now l_w (r / 300 nm)^gamma, gamma
rem fitted in the functional fit (direction_ms.length_fast_radius_exponent, t0.yaml functional_fit_length_radius: true),
rem explicit initial Nelder-Mead simplex, 800 evaluations (AA: 400 did not converge).
rem
rem AC1 refit of the direction model of the AA T0 run only (minutes-1 h): without gamma (same as AA1, new simplex) and with gamma,
rem     both evaluated on val per length x radius stratum. Look at kappa_autocorr_1um in the r<300 / r>=300 rows.
rem AC2 T0 end to end with the new t0.yaml + diagnosis + neuron CV on the same 5 folds as X2 / Y10 / AA.
rem     Criterion: kappa autocorrelation no longer significant in the CV, kappa rms and tortuosity stay non-significant.
rem Log: scripts\neuron_model\overnight\2026-11-19_AC_t0-length-radius.bat > logs_AC.txt 2>&1

set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set B=C:\Users\Student\LipsEkaterina\data\minnie65\preprocessed_branches
set SPLIT=data\processed\neuron-model\splits\splits_v1.parquet
set RUNS=runs\training\neuron-model
set T0AA=%RUNS%\2026-11-17_025725_t0

cd /d %REPO%

echo ===== [AC1] refit of the AA direction model: shared scale vs radius-dependent scale ===== %date% %time%
python scripts\neuron_model\diagnose_t0_direction.py --t0-run %T0AA% data.branches_root=%B% data.splits=%SPLIT% --refit --maxfev 800
python scripts\neuron_model\diagnose_t0_direction.py --t0-run %T0AA% data.branches_root=%B% data.splits=%SPLIT% --refit --maxfev 800 --length-radius

echo ===== [AC2] T0 with the radius-dependent wiggle scale + neuron CV ===== %date% %time%
python scripts\neuron_model\fit_t0.py --config configs\neuron-model\trunk\t0.yaml data.branches_root=%B% data.splits=%SPLIT% "model.mean_models=[parabola_mix]"
if errorlevel 1 (echo T0 FAILED & goto :end)
for /f "delims=" %%d in ('dir /b /ad /o-n %RUNS%\*_t0') do (set T0NEW=%RUNS%\%%d& goto :t0found)
:t0found
echo new T0 run: %T0NEW%
python scripts\neuron_model\diagnose_t0_direction.py --t0-run %T0NEW% data.branches_root=%B% data.splits=%SPLIT%
python scripts\neuron_model\cv_tp_neurons.py --branches-root %B% --splits %SPLIT% --k 5 --repeats 1 --models t0

:end
echo ===== chain AC done ===== %date% %time%
