@echo off
setlocal enabledelayedexpansion
rem 2026-11-26, chain AO1b. Repeat of step AO1 of chain AO (the quality report) after the fix of quality_report._tp_frames
rem (AO1 fell on the first T/P set: the evaluation table already had a "neuron" column, the join made it a duplicate,
rem TypeError: unhashable type 'numpy.ndarray'). CPU, ~15-20 min. Only reads existing outputs (AI, AH, AK0, AN + AO0).
rem T/P sets: newest AH sets (generated and real trunk, paired evaluation of AK0) and ALL sets of AN (seeds 1 and 2).
rem -> runs\analysis\neuron-model\2026-11-26_quality-report\quality_report.md (+ PNG, CSV); the old folder of the failed
rem    run is overwritten (only drs\ was written there).
rem Log: scripts\neuron_model\overnight\2026-11-26_AO1b_quality-report-rerun.bat > logs_AO1b.txt 2>&1

set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set ANA=runs\analysis\neuron-model
set EVAI=%ANA%\2026-11-21_045820_generator-evaluation-v3-vs-A-n500-2seeds
set TPH=%ANA%\tp_generation\minnie_h01_acfit
set TPN=%ANA%\tp_generation\minnie_h01_neuron
set QR=%ANA%\2026-11-26_quality-report

cd /d %REPO%
if not exist %EVAI%\real_test_morphometrics.parquet (echo NO evaluation of chain AI & goto :end)

echo ===== [AO1b] quality report ===== %date% %time%
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
if errorlevel 1 (echo QUALITY REPORT FAILED & goto :end)
dir /b %QR%

:end
echo ===== chain AO1b done ===== %date% %time%
goto :eof

:addnewest
for /f "delims=" %%d in ('dir /b /ad /o-n %1 2^>nul') do (
  if exist %1\%%d\evaluation_paired\evaluation_branches.csv set TP=!TP! %1\%%d@%2
  goto :eof
)
goto :eof
