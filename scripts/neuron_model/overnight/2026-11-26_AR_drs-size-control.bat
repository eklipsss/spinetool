@echo off
rem 2026-11-26, chain AR. Control of the density-ratio selection (chain AK) for sample size, on two generation seeds.
rem CPU only, ~8 h (most of it AR3). Independent of AN / AO / AL / AP / AQ (needs only the outputs of chains AI and AK);
rem can run next to the GPU chains, better not at the same time as another CPU-heavy evaluation (AO2, AP3).
rem
rem Why: in AK3 the selection removed the morphometric shift (7-8 -> 0 of 11 significant metrics, valid meshes 0.95), but the
rem Chamfer metrics of A got worse (coverage_cd 0.596 -> 0.553, mmd_cd 0.036 -> 0.052). The selected sets have ~225 samples
rem instead of 500, and the surface metrics use k = min(1000, n generated, n real) clouds per set, so they depend on n by
rem construction (AF: coverage of A 0.550 at n = 200, 0.602 at n = 500). Two readings:
rem   (1) only the smaller sample: a RANDOM subset of the same size is as bad as the selected one;
rem   (2) the selection narrows the shapes (keeps "average" spines by 11 numbers): the random subset is clearly better.
rem AK3 had no random subset in the full protocol and one generation seed only.
rem
rem AR1 density_ratio_selection.py as AK2 (selector fitted on TRAINING real spines of splits_v1 and generation seed 1, applied
rem     to seed 2, same --seed 0, so the kept set must be the same as in AK: A 226, v3 224); exports the kept samples AND a
rem     random subset of the same size -> runs\analysis\neuron-model\2026-11-26_drs-control\{A_drs,A_rand,v3_drs,v3_rand}\seed_2
rem AR2 the mirror: selector fitted on seed 2, applied to seed 1 -> ...\seed_1 (second independent generation sample)
rem AR3 evaluate_generators.py on seeds 1 and 2 (protocol of chains AI / AK3): A_drs, A_rand, v3_drs, v3_rand, real_poisson;
rem     paired neuron bootstrap (drs - rand) and the classifier ablation against real_poisson
rem     -> runs\analysis\neuron-model\<date>_generator-evaluation-drs-control
rem The full sets A and v3 on both seeds are already in the evaluation of chain AI (not repeated here).
rem Log: scripts\neuron_model\overnight\2026-11-26_AR_drs-size-control.bat > logs_AR.txt 2>&1

set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set SPLITS=data\processed\neuron-model\splits
set RUNS=runs\training\neuron-model
set ANA=runs\analysis\neuron-model
set V3=%RUNS%\2026-11-03_015237_mogen-conditional-v3
set MA=%RUNS%\2026-10-20_213103_mogen-ft-dev
set P3=%V3%\meshes_poisson_d8_pw8
set MAN=%MA%\meshes_poisson_d8_pw8_n500
set CMP2=%ANA%\2026-11-21_v3-vs-A-seeds
set EVAI=%ANA%\2026-11-21_045820_generator-evaluation-v3-vs-A-n500-2seeds
set PP=%EVAI%\postprocessed
set DRC=%ANA%\2026-11-26_drs-control
set FULL=C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/manifest_minnie65_full.parquet
set MORPH=C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/morphometrics_minnie65_full.parquet

cd /d %REPO%
if not exist %EVAI%\real_test_morphometrics.parquet (echo NO evaluation of chain AI & goto :end)
for %%f in (mogen_A_seed1 mogen_A_seed2 mogen_v3_seed1 mogen_v3_seed2 real_poisson_seed1 real_poisson_seed2) do (
  if not exist %PP%\morphometrics_%%f.parquet (echo NO %PP%\morphometrics_%%f.parquet & goto :end)
)
for %%s in (1 2) do (
  if not exist %MAN%\seed_%%s (echo NO A seed_%%s & goto :end)
  if not exist %P3%\seed_%%s (echo NO v3 seed_%%s & goto :end)
)
if exist %DRC%\A_drs (echo %DRC% already exists: move it to trash or rename it first & goto :end)

echo ===== [AR1] selection fitted on seed 1, applied to seed 2 (+ random subset of the same size) ===== %date% %time%
call :select A %MAN% 1 2
if errorlevel 1 goto :end
call :select v3 %P3% 1 2
if errorlevel 1 goto :end

echo ===== [AR2] selection fitted on seed 2, applied to seed 1 ===== %date% %time%
call :select A %MAN% 2 1
if errorlevel 1 goto :end
call :select v3 %P3% 2 1
if errorlevel 1 goto :end

echo ===== [AR3] evaluation on seeds 1 and 2 ===== %date% %time%
python scripts\neuron_model\evaluate_generators.py ^
  --model mogen_A_drs=%DRC%\A_drs --model mogen_A_rand=%DRC%\A_rand --model mogen_v3_drs=%DRC%\v3_drs --model mogen_v3_rand=%DRC%\v3_rand ^
  --model real_poisson=%CMP2%\real\poisson ^
  "data.manifests=['%FULL%']" data.path_mode=absolute data.splits=%SPLITS%\splits_v1_val_minus_dev.parquet ^
  data.bbox=%SPLITS%\bbox_splits_v1.json data.drop_unassigned=true "protocol.generation_seeds=[1,2]" real.morphometrics_table=%MORPH% ^
  real.chamfer_memorization=false name=generator-evaluation-drs-control
if errorlevel 1 (echo EVALUATION FAILED & goto :end)
for /f "delims=" %%d in ('dir /b /ad /o-n %ANA%\*_generator-evaluation-drs-control') do (set EVAL=%ANA%\%%d& goto :evalfound)
:evalfound
echo evaluation run: %EVAL%
python scripts\neuron_model\paired_generator_bootstrap.py --eval-run %EVAL% ^
  --models mogen_A_drs mogen_A_rand mogen_v3_drs mogen_v3_rand real_poisson ^
  --pairs mogen_A_drs:mogen_A_rand mogen_v3_drs:mogen_v3_rand mogen_A_drs:mogen_v3_drs mogen_A_drs:real_poisson mogen_v3_drs:real_poisson ^
  --n-boot 1000
python scripts\neuron_model\classifier_ablation.py --eval-run %EVAL% --reference real_poisson ^
  --models mogen_A_drs mogen_A_rand mogen_v3_drs mogen_v3_rand
type %EVAL%\postprocessed\summary.md

:end
echo ===== chain AR done ===== %date% %time%
goto :eof

rem :select <model> <set folder> <fit seed> <evaluation seed>
:select
python scripts\neuron_model\density_ratio_selection.py --real-fit %MORPH% --real-fit-splits %SPLITS%\splits_v1.parquet --real-fit-split train ^
  --gen-fit %PP%\morphometrics_mogen_%1_seed%3.parquet --real-eval %EVAI%\real_test_morphometrics.parquet ^
  --gen-eval %PP%\morphometrics_mogen_%1_seed%4.parquet --floor %PP%\morphometrics_real_poisson_seed%4.parquet ^
  --gen-eval-dir %2\seed_%4 --export %DRC%\%1_drs --export-random %DRC%\%1_rand --out %DRC%\report_%1_fit%3_eval%4
if errorlevel 1 (echo SELECTION %1 fit seed %3 FAILED & exit /b 1)
exit /b 0
