@echo off
setlocal enabledelayedexpansion
rem 2026-11-27, chain AS (for the pre-defence, slide 13.1). CPU only, ~30-60 min, reads existing outputs only (AI, AH, AN, AO0).
rem AS1: quality report again, now with the Kolmogorov-Smirnov test with the NEURON as the unit (significance.ks_neuron_floor_test:
rem      D of the model vs the distribution of D between two halves of the real neurons, 1000 splits) - new column in the T / P tables.
rem      -> runs\analysis\neuron-model\2026-11-27_quality-report-ks\quality_report.md (+ PNG, CSV); AO1b folder is not touched.
rem AS2: section 4 of the global report on the current T0 sets of Minnie65 (AH generated / real trunk, AN all modes and seeds):
rem      does the generator REPRODUCE the real differences between groups (compartment apical vs basal WITHIN a neuron, binary
rem      cell type between neurons) - real effect vs generated effect with bootstrap intervals over source neurons, verdicts
rem      consistent / same_sign_different_size / real_only / generated_only (global_report/generated.py, 4a), and equivalence
rem      within a group (4b). H01 is left out: 6 validation neurons with branches are too few for the within-neuron test.
rem      The real tables are copied from the newest global-tables folder WITHOUT its section-4 tables (they are not overwritten):
rem      -> runs\analysis\neuron-model\2026-11-27_global-tables-s4\generated_*.csv + <date>_global-report.html
rem Log: scripts\neuron_model\overnight\2026-11-27_AS_t-conditions-and-ks.bat > logs_AS.txt 2>&1

set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set ANA=runs\analysis\neuron-model
set EVAI=%ANA%\2026-11-21_045820_generator-evaluation-v3-vs-A-n500-2seeds
set TPH=%ANA%\tp_generation\minnie_h01_acfit
set TPN=%ANA%\tp_generation\minnie_h01_neuron
set QR=%ANA%\2026-11-27_quality-report-ks
set GTSRC=%ANA%\2026-11-17_042234_global-tables
set GT=%ANA%\2026-11-27_global-tables-s4

cd /d %REPO%

echo ===== [AS1] quality report with the neuron-level KS ===== %date% %time%
if not exist %EVAI%\real_test_morphometrics.parquet (echo NO evaluation of chain AI: AS1 skipped & goto :as2)
set TP=
for %%k in (minnie65 h01) do (
  for %%m in (generated real_t) do call :addnewest %TPH%\%%k\%%m AH-%%k-%%m
)
if exist %TPN% (
  for %%k in (minnie65 h01) do (
    for /f "delims=" %%d in ('dir /b /ad %TPN%\%%k\generated') do (
      if exist %TPN%\%%k\generated\%%d\evaluation\evaluation_branches.csv set TP=!TP! %TPN%\%%k\generated\%%d@AN-%%k-%%d
    )
  )
)
echo T/P sets: !TP!
python scripts\neuron_model\make_quality_report.py --out %QR% --s-eval %EVAI% ^
  --s-conditional %ANA%\2026-11-20_conditional-report-v4-mouse-valfloor %ANA%\2026-11-20_conditional-report-v4-types-valfloor %ANA%\2026-11-20_conditional-report-v4-human-valfloor ^
  --tp !TP!
if errorlevel 1 (echo QUALITY REPORT FAILED) else (dir /b %QR%)

:as2
echo ===== [AS2] section 4 of the global report on the current T0 sets (Minnie65) ===== %date% %time%
if not exist %GTSRC%\run_info.json (echo NO global tables %GTSRC%: AS2 skipped & goto :end)
if not exist %GT% (
  robocopy %GTSRC% %GT% /E /XF generated_* *global-report*.html /NFL /NDL /NJH /NJS /NP
  if errorlevel 8 (echo COPY OF THE GLOBAL TABLES FAILED & goto :end)
)
set GEN=
call :addnewest2 %TPH%\minnie65\generated AH-generated
call :addnewest2 %TPH%\minnie65\real_t AH-real_t
if exist %TPN%\minnie65\generated (
  for /f "delims=" %%d in ('dir /b /ad %TPN%\minnie65\generated') do set GEN=!GEN! --generated-tp %TPN%\minnie65\generated\%%d@AN-%%d
)
echo section-4 sets: !GEN!
python scripts\neuron_model\build_global_tables.py --add-to %GT% !GEN!
if errorlevel 1 (echo SECTION 4 FAILED & goto :end)
python scripts\neuron_model\build_global_report.py --dir %GT%
dir /b %GT%\generated_*

:end
echo ===== chain AS done ===== %date% %time%
goto :eof

:addnewest
for /f "delims=" %%d in ('dir /b /ad /o-n %1 2^>nul') do (
  if exist %1\%%d\evaluation_paired\evaluation_branches.csv set TP=!TP! %1\%%d@%2
  goto :eof
)
goto :eof

:addnewest2
for /f "delims=" %%d in ('dir /b /ad /o-n %1 2^>nul') do (
  if exist %1\%%d\generation_info.json set GEN=!GEN! --generated-tp %1\%%d@%2
  goto :eof
)
goto :eof
