@echo off
setlocal enabledelayedexpansion
rem 2026-11-27, chain AU (CPU, ~20-60 min, reads only tables and the generation_metadata.json of the real controls).
rem Kolmogorov-Smirnov with the NEURON as the unit for the 11 spine morphometrics (scripts\neuron_model\ks_spine_models.py,
rem significance.ks_sample_vs_neurons): real after the chain vs real, model vs real, model vs real after the same chain.
rem AU1 unconditional evaluation of chain AI (val v1 minus dev, slides 13.3): v3 and A, floor real_poisson (d8_pw8)
rem AU2 final evaluation of chain S (untouched test of splits_dev_100n): VAE_flow with its floor real_mc, A with real_poisson
rem AU3 conditional v4 by organism x binary cell type (chain AH report), floor real-pw8val, real val spines from the stage-8 tables
rem -> runs\analysis\neuron-model\2026-11-27_ks-spine-models\{ai,final,v4}\ks_summary.md (+ ks_<set>.csv)
rem Log: scripts\neuron_model\overnight\2026-11-27_AU_ks-spine-models.bat > logs_AU.txt 2>&1

set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set ANA=runs\analysis\neuron-model
set OUT=%ANA%\2026-11-27_ks-spine-models
cd /d %REPO%

echo ===== [AU1] AI: v3 and A ===== %date% %time%
set EVAI=%ANA%\2026-11-21_045820_generator-evaluation-v3-vs-A-n500-2seeds
if exist %EVAI%\real_test_morphometrics.parquet (
  python scripts\neuron_model\ks_spine_models.py --eval-run %EVAI% --model mogen_v3:real_poisson --model mogen_A:real_poisson --out %OUT%\ai
) else (echo NO evaluation %EVAI%)

echo ===== [AU2] final evaluation: VAE_flow and A ===== %date% %time%
set EVS=%ANA%\2026-11-10_010917_generator-evaluation-final
if exist %EVS%\real_test_morphometrics.parquet (
  python scripts\neuron_model\ks_spine_models.py --eval-run %EVS% --model VAE_flow:real_mc --model mogen_A:real_poisson --out %OUT%\final
) else (echo NO evaluation %EVS%)

echo ===== [AU3] v4 by organism and binary cell type ===== %date% %time%
set SPG=
for /f "delims=" %%d in ('dir /b /ad /o-n %ANA%\*species-generation-v4-valfloor 2^>nul') do (if "!SPG!"=="" set SPG=%ANA%\%%d)
if "%SPG%"=="" (echo NO species-generation-v4-valfloor report & goto :end)
python scripts\neuron_model\ks_spine_models.py --species-report %SPG% --out %OUT%\v4

:end
echo ===== chain AU done ===== %date% %time%
