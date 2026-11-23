@echo off
rem 2026-11-22: move finished / superseded run folders from C: to the NAS (C: ran out of space in chain AJ).
rem Kept on C: everything the next weeks need: MoGen A, A curves, v3, v4, VAE base, current T0 / P0 / P1,
rem webapp models, all analysis folders from 2026-11-17 on, compartment tables, figures, tp_generation, caches.
rem
rem   scripts\neuron_model\maintenance\2026-11-22_move_old_runs_to_nas.bat sizes   -> only print the size of every listed folder
rem   scripts\neuron_model\maintenance\2026-11-22_move_old_runs_to_nas.bat move    -> robocopy /MOVE to %NAS% (log move_runs_log.txt)
rem robocopy /MOVE deletes a source file only after it was copied; a folder whose copy failed stays on C: (see the log).
rem Moving back: the same robocopy with source and target swapped.

set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set NAS=O:\neuron-model\runs
set LOG=%REPO%\move_runs_log.txt

set ANA_LIST=2026-10-08_012425_reconstruction-calibration 2026-10-08_013446_reconstruction-calibration 2026-10-08_020336_reconstruction-calibration ^
 2026-10-08_032710_reconstruction-calibration 2026-10-08_035319_reconstruction-calibration 2026-10-08_035540_reconstruction-calibration ^
 2026-10-08_051336_reconstruction-calibration 2026-10-21_000926_mogen-zero-shot 2026-10-21_234831_generator-evaluation 2026-10-22_031920_generator-evaluation ^
 2026-10-28_024345_condition-effects 2026-10-28_024645_condition-effects 2026-11-01_180529_morphology-by-condition 2026-11-02_170928_branch-stats-by-condition ^
 2026-11-02_192038_apical-by-direction-minnie65 2026-11-02_195824_apical-by-direction-minnie65 2026-11-03_005059_apical-by-direction-minnie65 ^
 2026-11-03_031727_generator-evaluation 2026-11-03_084700_conditional-generation 2026-11-03_092039_conditional-generation 2026-11-03_160122_conditional-generation ^
 2026-11-03_163357_conditional-generation 2026-11-04_022355_global-tables 2026-11-04_024519_global-tables 2026-11-04_025252_global-tables global-tables-probe ^
 2026-11-04_094722_conditional-report 2026-11-04_191049_conditional-report 2026-11-06_004118_real-controls 2026-11-06_010305_generator-evaluation ^
 2026-11-06_014629_global-tables_attached-only 2026-11-06_112409_generator-evaluation-untouched 2026-11-06_real-controls-r256 ^
 2026-11-07_085802_vae-decoder-study 2026-11-07_142955_vae-decoder-study 2026-11-07_195048_vae-decoder-study 2026-11-08_011445_generator-evaluation-zopt ^
 2026-11-08_015641_vae-decoder-study 2026-11-08_180137_generator-evaluation-anchored-val 2026-11-08_212101_generator-evaluation-autodecoder-val ^
 2026-11-08_214603_vae-decoder-study 2026-11-08_real-controls-val 2026-11-09_013155_poisson-sweep-val 2026-11-09_103837_generator-evaluation-beta-val ^
 2026-11-09_111639_vae-decoder-study 2026-11-09_114917_mogen-poisson-val 2026-11-09_real-controls-poisson-final 2026-11-09_real-poisson-sweep ^
 2026-11-10_010917_generator-evaluation-final 2026-11-10_020046_axon-rule-check 2026-11-14_023444_tp-neuron-cv 2026-11-15_080459_tp-neuron-cv ^
 2026-11-15_real-controls-by-type 2026-11-16_conditional-report-pw8-binary 2026-11-16_conditional-report-pw8-types 2026-11-16_real-controls-by-type ^
 2026-11-18_020108_generator-evaluation-v3-vs-A 2026-11-18_v3-vs-A 2026-11-18_212234_generator-evaluation-v3-vs-A-n500 ^
 2026-11-19_conditional-report-v4-human 2026-11-19_conditional-report-v4-mouse 2026-11-19_real-controls-h01 2026-11-19_species-generation-v4

set TRN_LIST=2026-10-08_012455_mogen-pretrained-finetune 2026-10-08_021150_vae-baseline 2026-10-14_004155_vae-dev-smoke 2026-10-14_005229_vae-dev-smoke24 ^
 2026-10-14_031841_vae-dev-fix 2026-10-14_032414_vae-dev-fix 2026-10-14_125835_vae-dev-bn 2026-10-14_171150_vae-dev-main 2026-10-15_234133_vae-dev-reg ^
 2026-10-17_232134_vae-dev-reg0 2026-10-18_204218_vae-dev-reg0-w512 2026-10-20_024842_p0 2026-10-20_133113_t0 2026-10-20_133132_p0 2026-10-20_142253_t0 ^
 2026-10-20_142311_p0 2026-10-20_142946_t0 2026-10-20_143003_p0 2026-10-20_151634_t0 2026-10-20_195005_t0 2026-10-20_200115_t0 2026-10-21_001206_mogen-scratch-dev ^
 2026-10-21_001448_mogen-scratch-dev 2026-10-24_152229_t0 2026-10-27_012844_t0 2026-10-28_003441_p0 2026-10-28_014059_t0 2026-10-28_020020_t0 2026-10-28_022026_t0 ^
 2026-10-28_025253_mogen-conditional 2026-10-28_231647_t0 2026-10-29_002016_p0 2026-10-29_005404_p0 2026-10-29_052942_t0 2026-10-29_054617_p0 2026-10-29_063649_t0 ^
 2026-11-01_061733_mogen-conditional-v1 2026-11-02_020009_t0 2026-11-02_023219_mogen-conditional-v2 2026-11-03_015114_mogen-conditional-v3 2026-11-03_113031_t0 ^
 2026-11-03_203955_p0 2026-11-04_015910_mogen-scratch-dev-curves 2026-11-04_015937_vae-dev-reg0-cos-curves 2026-11-04_170910_mogen-scratch-dev-curves ^
 2026-11-07_013735_vae-dev-beta1e-5 2026-11-07_040420_vae-dev-normal0.1 2026-11-07_063137_vae-dev-omega60 2026-11-08_200626_vae-autodecoder-n0 ^
 2026-11-08_201537_vae-autodecoder-n1 2026-11-09_012238_vae-dev-beta3.0e-4 2026-11-09_013719_p1 2026-11-09_042912_vae-dev-beta1.0e-3 2026-11-09_073128_vae-dev-beta3.0e-3 ^
 2026-11-10_113612_t0 2026-11-10_113713_p0 2026-11-10_114732_p1 2026-11-11_004830_p0 2026-11-11_005857_p1 2026-11-11_102511_p0 2026-11-11_103532_p1 2026-11-15_044148_t0

cd /d %REPO%
if "%1"=="sizes" goto :sizes
if "%1"=="move" goto :move
echo usage: %~nx0 sizes ^| move
goto :eof

:sizes
for %%d in (%ANA_LIST%) do call :size runs\analysis\neuron-model\%%d
for %%d in (%TRN_LIST%) do call :size runs\training\neuron-model\%%d
powershell -NoProfile -Command "$t=0; foreach($l in Get-Content '%REPO%\move_runs_sizes.txt'){ $t+=[double]($l.Split(' ')[0]) }; 'TOTAL ' + [int64]($t/1024) + ' GB'"
del %REPO%\move_runs_sizes.txt
goto :eof

:size
if not exist "%~1" (echo MISSING %~1 & goto :eof)
for /f %%s in ('powershell -NoProfile -Command "[int64](((Get-ChildItem -LiteralPath '%~1' -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum) / 1MB)"') do (echo %%s MB  %~1 & echo %%s>> %REPO%\move_runs_sizes.txt)
goto :eof

:move
if not exist O:\ (echo NAS drive O: not available & goto :eof)
echo ===== move to %NAS% ===== %date% %time% >> %LOG%
for %%d in (%ANA_LIST%) do call :mv analysis\neuron-model\%%d
for %%d in (%TRN_LIST%) do call :mv training\neuron-model\%%d
echo ===== done ===== %date% %time% >> %LOG%
echo done, see %LOG%
goto :eof

:mv
if not exist "runs\%~1" (echo MISSING runs\%~1 >> %LOG% & goto :eof)
echo %time% moving runs\%~1
robocopy "runs\%~1" "%NAS%\%~1" /E /MOVE /R:2 /W:5 /NP /NFL /NDL /NJH >> %LOG%
if errorlevel 8 (echo FAILED runs\%~1 - left on C: >> %LOG% & echo FAILED runs\%~1) else (echo OK runs\%~1 >> %LOG%)
goto :eof
