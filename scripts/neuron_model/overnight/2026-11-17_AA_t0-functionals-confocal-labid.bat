@echo off
rem 2026-11-17, chain AA (CPU only, ~5-7 h). Run after git pull.
rem
rem Part A, T0 bending (chain Y: the pooled semivariograms matched, the per-branch functionals did not):
rem AA1 diagnosis of the CURRENT direction model (chain Y T0): one chain per train / val branch with its REAL radius and length,
rem     kappa rms, tortuosity, kappa autocorrelation per length x radius stratum; then --refit to the per-branch functional
rem     quantiles (direction_functional.py) and the refit model on val. Minutes.
rem AA2 T0 refit end to end with model.direction_options.fit_target=functionals (t0.yaml since 2026-11-17; the semivariogram
rem     fit without simulation corrections is only the start) + neuron cross-validation of T0 on the same 5 folds as X2 / Y10.
rem     Criterion: no significant discrepancy of kappa rms, tortuosity, kappa autocorrelation in the CV.
rem
rem Part B, confocal imaging model (docs/neuron-model/confocal-imaging-model.md; parameters unknown until the laboratory answers):
rem AA3 Minnie65 train spines (local cache) as they are + through the imaging model on a grid: PSF narrow (55 / 180 nm, NA 1.4
rem     confocal) and wide (90 / 270 nm), threshold 0.2 / 0.3 / 0.5, with the trunk; one isolated-spine variant; LabID healthy
rem     (group Wt, z 0.7) as they are; decomposition of the Minnie65 -> LabID difference into imaging and rest.
rem
rem Part C, LabID in the global report:
rem AA4 global tables again with the LabID BRANCHES added (2026-11-06 run had LabID spines only, no T / P rows for health and
rem     treatment): density, empty share, g(r) along the branch, interval CV, neighbour distance, mark correlations by
rem     health | treatment. Unit = branch folder (animals unknown).
rem Log: scripts\neuron_model\overnight\2026-11-17_AA_t0-functionals-confocal-labid.bat > logs_AA.txt 2>&1

set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set B=C:\Users\Student\LipsEkaterina\data\minnie65\preprocessed_branches
set SPLIT=data\processed\neuron-model\splits\splits_v1.parquet
set RUNS=runs\training\neuron-model
set ANA=runs\analysis\neuron-model
set T0Y=%RUNS%\2026-11-15_044148_t0
set GT=%ANA%\2026-11-06_013458_global-tables
set CACHE=cache\minnie65_v1_60k\manifest.parquet
set LABID_MAN=C:\Users\Student\LipsEkaterina\data\labid\labid_raw_z07\labid\preprocessed\manifest.parquet
set LABID_BR=C:\Users\Student\LipsEkaterina\data\labid\preprocessed_branches_z07
set CONF=%ANA%\2026-11-17_confocal

cd /d %REPO%

rem ================================================================ part A
echo ===== [AA1] direction model of chain Y on train / val + refit to the functionals ===== %date% %time%
python scripts\neuron_model\diagnose_t0_direction.py --t0-run %T0Y% data.branches_root=%B% data.splits=%SPLIT% --refit

echo ===== [AA2] T0 with the functional fit + neuron CV ===== %date% %time%
python scripts\neuron_model\fit_t0.py --config configs\neuron-model\trunk\t0.yaml data.branches_root=%B% data.splits=%SPLIT% "model.mean_models=[parabola_mix]"
if errorlevel 1 (echo T0 FAILED & goto :partb)
for /f "delims=" %%d in ('dir /b /ad /o-n %RUNS%\*_t0') do (set T0NEW=%RUNS%\%%d& goto :t0found)
:t0found
echo new T0 run: %T0NEW%
python scripts\neuron_model\diagnose_t0_direction.py --t0-run %T0NEW% data.branches_root=%B% data.splits=%SPLIT%
python scripts\neuron_model\cv_tp_neurons.py --branches-root %B% --splits %SPLIT% --k 5 --repeats 1 --models t0

rem ================================================================ part B
:partb
echo ===== [AA3] confocal imaging model: Minnie65 train spines, LabID Wt ===== %date% %time%
if not exist %CACHE% (echo MISSING %CACHE% & goto :partc)
python scripts\neuron_model\make_real_controls.py --manifest %CACHE% --splits %SPLIT% --split train --controls asis confocal ^
  --confocal-param sigma_xy_nm=55 --confocal-param sigma_z_nm=180 --confocal-param threshold=0.3 --n-samples 500 --workers 8 --out %CONF%\minnie_n_t03
for %%t in (0.2 0.5) do (
  python scripts\neuron_model\make_real_controls.py --manifest %CACHE% --splits %SPLIT% --split train --controls confocal ^
    --confocal-param sigma_xy_nm=55 --confocal-param sigma_z_nm=180 --confocal-param threshold=%%t --n-samples 500 --workers 8 --out %CONF%\minnie_n_t%%t
)
for %%t in (0.2 0.3 0.5) do (
  python scripts\neuron_model\make_real_controls.py --manifest %CACHE% --splits %SPLIT% --split train --controls confocal ^
    --confocal-param sigma_xy_nm=90 --confocal-param sigma_z_nm=270 --confocal-param threshold=%%t --n-samples 500 --workers 8 --out %CONF%\minnie_w_t%%t
)
python scripts\neuron_model\make_real_controls.py --manifest %CACHE% --splits %SPLIT% --split train --controls confocal ^
  --confocal-param sigma_xy_nm=55 --confocal-param sigma_z_nm=180 --confocal-param threshold=0.3 --confocal-param trunk_radius_nm=null --n-samples 500 --workers 8 --out %CONF%\minnie_n_t03_isolated
python scripts\neuron_model\make_real_controls.py --manifest %LABID_MAN% --where "neuron_id~^Wt_[0-9]" --controls asis --n-samples 250 --workers 8 --out %CONF%\labid_wt
python scripts\neuron_model\confocal_domain_shift.py --reference minnie=%CONF%\minnie_n_t03\asis\seed_1 --target labid_wt=%CONF%\labid_wt\asis\seed_1 ^
  --imaged n_t0.2=%CONF%\minnie_n_t0.2\confocal\seed_1 --imaged n_t0.3=%CONF%\minnie_n_t03\confocal\seed_1 --imaged n_t0.5=%CONF%\minnie_n_t0.5\confocal\seed_1 ^
  --imaged w_t0.2=%CONF%\minnie_w_t0.2\confocal\seed_1 --imaged w_t0.3=%CONF%\minnie_w_t0.3\confocal\seed_1 --imaged w_t0.5=%CONF%\minnie_w_t0.5\confocal\seed_1 ^
  --imaged n_t0.3_isolated=%CONF%\minnie_n_t03_isolated\confocal\seed_1 --out %CONF%\shift

rem ================================================================ part C
:partc
echo ===== [AA4] global tables with the LabID branches ===== %date% %time%
python scripts\neuron_model\build_global_tables.py --from-run %GT% --extra-branches labid=%LABID_BR%
for /f "delims=" %%d in ('dir /b /ad /o-n %ANA%\*_global-tables') do (set GTNEW=%ANA%\%%d& goto :gtfound)
:gtfound
python scripts\neuron_model\build_global_report.py --dir %GTNEW%

echo ===== AA done ===== %date% %time%
