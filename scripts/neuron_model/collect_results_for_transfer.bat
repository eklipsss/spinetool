@echo off
rem 2026-11-19. Collects the small result files (tables, json, reports, logs) listed in
rem docs/neuron-model/sprints/19.11-conditions-inventory-and-transfer.md into ONE folder and zips it, for copying to the mac.
rem Meshes, point clouds, checkpoints and any file above 300 MB are NOT copied. Nothing in the repository is changed or deleted.
rem Run on Windows from anywhere:   scripts\neuron_model\collect_results_for_transfer.bat > collect_log.txt 2>&1
rem Afterwards read MISSING.txt in the destination folder: every line is a folder or file that was not found and must be looked up by hand.

set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set DST=C:\Users\Student\LipsEkaterina\transfer_2026-11-19
rem LabID: the converted trees with the S outputs (z07 = z correction 0.7 for Wt and Ab, z1 = no correction); manifest and morphometrics are small
set LABID07=C:\Users\Student\LipsEkaterina\data\labid\labid_raw_z07\labid\preprocessed
set LABID1=C:\Users\Student\LipsEkaterina\data\labid\labid_raw_z1\labid\preprocessed

set TRN=%REPO%\runs\training\neuron-model
set ANA=%REPO%\runs\analysis\neuron-model
set HEAVY=*.off *.ply *.npy *.npz *.pt *.pth *.ckpt *.safetensors *.tif *.tiff
set LOG=%DST%\MISSING.txt

if not exist "%REPO%" (echo REPO NOT FOUND: %REPO% & goto :end)
if not exist "%DST%" mkdir "%DST%"
echo missing or unchecked items (%date% %time%) > "%LOG%"

echo ===== A. training runs: only small files (run_info, configs, logs) =====
for %%R in (2026-10-20_213103_mogen-ft-dev 2026-11-02_023219_mogen-conditional-v2 2026-11-03_015237_mogen-conditional-v3 2026-11-17_085545_mogen-conditional-v4-h01 2026-11-01_061733_mogen-conditional-v1) do call :run_small %%R

echo ===== B. generated sets of A and v3: json descriptions only =====
call :json_only "%TRN%\2026-10-20_213103_mogen-ft-dev\meshes_poisson_d8_pw8_n500" training\2026-10-20_213103_mogen-ft-dev\meshes_poisson_d8_pw8_n500
call :json_only "%TRN%\2026-10-20_213103_mogen-ft-dev\meshes_poisson_d8_pw8" training\2026-10-20_213103_mogen-ft-dev\meshes_poisson_d8_pw8
call :json_only "%TRN%\2026-11-03_015237_mogen-conditional-v3\meshes_poisson_d8_pw8" training\2026-11-03_015237_mogen-conditional-v3\meshes_poisson_d8_pw8

echo ===== C. evaluation of A and v3 =====
call :full "%ANA%\2026-11-18_212234_generator-evaluation-v3-vs-A-n500" analysis\2026-11-18_212234_generator-evaluation-v3-vs-A-n500
call :full "%ANA%\2026-11-18_020108_generator-evaluation-v3-vs-A" analysis\2026-11-18_020108_generator-evaluation-v3-vs-A
call :full "%ANA%\2026-11-18_v3-vs-A\real" analysis\2026-11-18_v3-vs-A\real

echo ===== D. real spines by condition, global tables =====
call :full "%ANA%\2026-11-01_184009_morphology-by-condition" analysis\2026-11-01_184009_morphology-by-condition
call :full "%ANA%\2026-11-06_013458_global-tables" analysis\2026-11-06_013458_global-tables
call :full "%ANA%\2026-11-06_014629_global-tables_attached-only" analysis\2026-11-06_014629_global-tables_attached-only
rem the newest global-tables folder by name (date prefix); printed so that you can check its date
set GT=
for /f "delims=" %%d in ('dir /b /ad /o-n "%ANA%\*_global-tables" 2^>nul') do if not defined GT set GT=%%d
if defined GT (echo newest global-tables: %GT% & call :full "%ANA%\%GT%" analysis\%GT%) else (echo no global-tables folder found >> "%LOG%")
rem html reports lying next to the folders (top level of the analysis folder)
if not exist "%DST%\analysis" mkdir "%DST%\analysis"
robocopy "%ANA%" "%DST%\analysis" *global-report*.html *conditional-generation-report*.html /NFL /NDL /NJH /NJS /NP >nul

echo ===== E. conditional generation reports =====
call :full "%ANA%\2026-11-16_conditional-report-pw8-binary" analysis\2026-11-16_conditional-report-pw8-binary
call :full "%ANA%\2026-11-16_conditional-report-pw8-types" analysis\2026-11-16_conditional-report-pw8-types
call :full "%ANA%\2026-11-15_conditional-report-pw8-binary" analysis\2026-11-15_conditional-report-pw8-binary
call :full "%ANA%\2026-11-15_conditional-report-pw8-types" analysis\2026-11-15_conditional-report-pw8-types
call :full "%ANA%\2026-11-04_094722_conditional-report" analysis\2026-11-04_094722_conditional-report
call :full "%ANA%\2026-11-16_real-controls-by-type" analysis\2026-11-16_real-controls-by-type

echo ===== F. check_condition_use results (location is not certain: searched by name) =====
for %%R in (2026-11-01_061733_mogen-conditional-v1 2026-11-02_023219_mogen-conditional-v2 2026-11-03_015237_mogen-conditional-v3) do (
  if exist "%TRN%\%%R" robocopy "%TRN%\%%R" "%DST%\training\%%R" *condition_use* /S /NFL /NDL /NJH /NJS /NP >nul
)
for /d %%D in ("%ANA%\*condition-use*") do robocopy "%%D" "%DST%\analysis\%%~nxD" /E /XF %HEAVY% /NFL /NDL /NJH /NJS /NP >nul

echo ===== G. LabID manifests and morphometrics (z07 and z1) =====
for %%V in (07 1) do call :labid %%V

echo ===== size of the result and zip =====
powershell -NoProfile -Command "$s=(Get-ChildItem '%DST%' -Recurse -File | Measure-Object Length -Sum).Sum/1MB; Write-Host ('total size, MB: ' + [math]::Round($s,1)); Get-ChildItem '%DST%' -Recurse -File | Sort-Object Length -Descending | Select-Object -First 8 | ForEach-Object { Write-Host ([math]::Round($_.Length/1MB,1).ToString() + ' MB  ' + $_.FullName) }"
powershell -NoProfile -Command "Compress-Archive -Path '%DST%\*' -DestinationPath '%DST%.zip' -Force"
echo zip: %DST%.zip
echo.
echo ===== now open %LOG% and look up by hand everything listed there =====
type "%LOG%"
goto :end

rem ---- helpers ---------------------------------------------------------------------------------------------------
:labid
rem %1 = 07 or 1; copies manifest.parquet and morphometrics.parquet of the LabID S outputs
if "%1"=="07" (set LSRC=%LABID07%) else (set LSRC=%LABID1%)
if not exist "%LSRC%" (echo MISSING LabID folder: %LSRC% >> "%LOG%" & goto :eof)
if not exist "%DST%\labid_z%1" mkdir "%DST%\labid_z%1"
for %%F in (manifest.parquet morphometrics.parquet metadata.parquet) do (
  if exist "%LSRC%\%%F" (copy /Y "%LSRC%\%%F" "%DST%\labid_z%1\%%F" >nul) else (echo MISSING LabID file: %LSRC%\%%F >> "%LOG%")
)
goto :eof

:run_small
rem %1 = run folder name in runs\training\neuron-model; copies top-level json/yaml/yml/txt, logs\*.jsonl, *.log
if not exist "%TRN%\%1" (echo MISSING run folder: %TRN%\%1 >> "%LOG%" & goto :eof)
robocopy "%TRN%\%1" "%DST%\training\%1" *.json *.yaml *.yml *.txt *.md /NFL /NDL /NJH /NJS /NP >nul
if exist "%TRN%\%1\logs" robocopy "%TRN%\%1\logs" "%DST%\training\%1\logs" *.jsonl *.txt *.log *.json /NFL /NDL /NJH /NJS /NP >nul
if not exist "%TRN%\%1\logs\metrics_val.jsonl" echo NOTE: %1 has no logs\metrics_val.jsonl, validation loss must be taken from the training log by hand >> "%LOG%"
goto :eof

:json_only
rem %1 = source folder, %2 = destination relative to %DST%; only *.json and small tables, recursive
if not exist %1 (echo MISSING folder: %~1 >> "%LOG%" & goto :eof)
robocopy %1 "%DST%\%~2" *.json *.csv /S /NFL /NDL /NJH /NJS /NP >nul
goto :eof

:full
rem %1 = source folder, %2 = destination relative to %DST%; everything except heavy file types and files above 300 MB
if not exist %1 (echo MISSING folder: %~1 >> "%LOG%" & goto :eof)
robocopy %1 "%DST%\%~2" /E /XF %HEAVY% /XD checkpoints /MAX:300000000 /NFL /NDL /NJH /NJS /NP >nul
goto :eof

:end
echo ===== done ===== %date% %time%
