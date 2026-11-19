@echo off
rem 2026-11-19, chain AG: evaluation of the SPECIES condition of MoGen v4 (GPU generation ~1.5 h, CPU in parallel, reports ~1 h; ~3 h in total).
rem Run after git pull, independent of the other chains (AE also generates v4 sets, see AE3: if its v4_*_g1.25_seed_1 folders already exist, the
rem generation below is skipped for them).
rem
rem What is evaluated: v4 generated with species x binary cell type (guidance 1.25, 300 per set, seed 1, d8_pw8 Poisson chain) against the REAL
rem spines of the same group (mouse = Minnie65, human = H01), validation neurons of splits_v1_h01 (neither v4 nor v3 saw them). Real spines through the
rem same chain are the floor (real-pw8): Minnie65 controls of chain Z are reused, H01 controls are made here (500 train spines per type, local cache
rem cache\h01_v1, in the background while the GPU generates). H01 spines without a neck (h01_detached_spines.parquet) are left out of the real side,
rem as in the v4 training.
rem AG1 generation + Poisson          AG2 H01 real controls (background)
rem AG3 compare_species_generation.py (species effect inside a type, type effect inside a species: ratio of medians, generated vs real)
rem AG4 conditional_generation_report.py, mouse (v4 and v3 sets, no-regression check on Minnie65) and human (H01, only v4)
rem NOT done here (after the report): detailed types, all conditions, the compartment (not a v4 block).
rem Log: scripts\neuron_model\overnight\2026-11-19_AG_v4-species-evaluation.bat > logs_AG.txt 2>&1

set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set SPLITS=data\processed\neuron-model\splits
set SPLITH=%SPLITS%\splits_v1_h01.parquet
set DETACHED=%SPLITS%\h01_detached_spines.parquet
set RUNS=runs\training\neuron-model
set ANA=runs\analysis\neuron-model
set CFG=configs\neuron-model\reconstruction\poisson_sweep
set V3=%RUNS%\2026-11-03_015237_mogen-conditional-v3
set V4=%RUNS%\2026-11-17_085545_mogen-conditional-v4-h01
set P4=%V4%\meshes_poisson_d8_pw8
set P3=%V3%\meshes_poisson_d8_pw8
set CTRLM=%ANA%\2026-11-16_real-controls-by-type\d8_pw8
set CTRLH=%ANA%\2026-11-19_real-controls-h01\d8_pw8
set HMAN=cache\h01_v1\manifest.parquet
set MORPHM=C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/morphometrics_minnie65_full.parquet
set MORPHH=C:/Users/Student/LipsEkaterina/data/h01/preprocessed_v2/H01_Seg/morphometrics.parquet
set V3E=cond_cell_type_binary-excitatory_g1.25_seed_1_n500
set V3I=cond_cell_type_binary-inhibitory_g1.25_seed_1_n500

cd /d %REPO%
if "%1"=="realctrl" goto :realctrl
if not exist %V4% (echo NO v4 RUN & goto :end)
if not exist %HMAN% (echo NO H01 CACHE %HMAN% & goto :end)
if not exist %CTRLM%\excitatory\poisson\seed_1 (echo NO Minnie65 REAL CONTROLS of chain Z & goto :end)
if not exist %MORPHH% (echo NO H01 morphometrics table %MORPHH% & goto :end)
if not exist %CTRLH% mkdir %CTRLH%
if exist %CTRLH%\_done.txt del %CTRLH%\_done.txt

echo ===== [AG2] H01 real controls in the background ===== %date% %time%
start "AG_realctrl" /b cmd /c ""%~f0" realctrl > logs_AG_realctrl.txt 2>&1"

echo ===== [AG1] MoGen v4: species x binary type, 300 each, guidance 1.25 (GPU) ===== %date% %time%
for %%s in (mouse human) do (
  for %%c in (excitatory inhibitory) do (
    if exist %P4%\v4_%%s_%%c_g1.25_seed_1\sample_00299 (echo set v4_%%s_%%c already reconstructed, skipped) else (
      python scripts\neuron_model\generate_mogen.py --run %V4% --checkpoint latest --condition species=%%s,cell_type_binary=%%c --guidance 1.25 --n-samples 300 --seed 1 --tag v4_%%s_%%c_g1.25_seed_1 --no-mesh
      if errorlevel 1 (echo GENERATION %%s %%c FAILED) else (
        python scripts\neuron_model\reconstruct_mogen_poisson.py --run %V4% --set v4_%%s_%%c_g1.25_seed_1 --out %P4% --poisson-config %CFG%\d8_pw8.yaml --workers 8
      )
    )
  )
)

echo ===== waiting for the H01 real controls ===== %date% %time%
:waitreal
if exist %CTRLH%\_done.txt goto :realdone
ping -n 61 127.0.0.1 >nul
goto :waitreal
:realdone
echo H01 controls done %date% %time%

echo ===== [AG3] species effect inside a type and type effect inside a species (ratios of medians) ===== %date% %time%
python scripts\neuron_model\compare_species_generation.py ^
  --gen mouse/excitatory@v4-pw8=%P4%\v4_mouse_excitatory_g1.25_seed_1 --gen mouse/inhibitory@v4-pw8=%P4%\v4_mouse_inhibitory_g1.25_seed_1 ^
  --gen human/excitatory@v4-pw8=%P4%\v4_human_excitatory_g1.25_seed_1 --gen human/inhibitory@v4-pw8=%P4%\v4_human_inhibitory_g1.25_seed_1 ^
  --gen mouse/excitatory@real-pw8=%CTRLM%\excitatory\poisson\seed_1 --gen mouse/inhibitory@real-pw8=%CTRLM%\inhibitory\poisson\seed_1 ^
  --gen human/excitatory@real-pw8=%CTRLH%\excitatory\poisson\seed_1 --gen human/inhibitory@real-pw8=%CTRLH%\inhibitory\poisson\seed_1 ^
  --minnie-table %MORPHM% --h01-table %MORPHH% --h01-detached %DETACHED% --splits %SPLITH% --split val ^
  --out %ANA%\2026-11-19_species-generation-v4

echo ===== [AG4a] conditional report, MOUSE (Minnie65; v4 and v3 sets, real-pw8 floor) ===== %date% %time%
python scripts\neuron_model\conditional_generation_report.py --block cell_type_binary --datasets minnie65 --morphometrics-table %MORPHM% ^
  --splits %SPLITH% --split val ^
  --gen excitatory@v4-pw8=%P4%\v4_mouse_excitatory_g1.25_seed_1 --gen inhibitory@v4-pw8=%P4%\v4_mouse_inhibitory_g1.25_seed_1 ^
  --gen excitatory@v3-pw8=%P3%\%V3E% --gen inhibitory@v3-pw8=%P3%\%V3I% ^
  --gen excitatory@real-pw8=%CTRLM%\excitatory\poisson\seed_1 --gen inhibitory@real-pw8=%CTRLM%\inhibitory\poisson\seed_1 ^
  --out %ANA%\2026-11-19_conditional-report-v4-mouse

echo ===== [AG4b] conditional report, HUMAN (H01; v4 sets, real-pw8 floor) ===== %date% %time%
python scripts\neuron_model\conditional_generation_report.py --block cell_type_binary --datasets h01 --morphometrics-table %MORPHH% ^
  --exclude-keys %DETACHED% --splits %SPLITH% --split val ^
  --gen excitatory@v4-pw8=%P4%\v4_human_excitatory_g1.25_seed_1 --gen inhibitory@v4-pw8=%P4%\v4_human_inhibitory_g1.25_seed_1 ^
  --gen excitatory@real-pw8=%CTRLH%\excitatory\poisson\seed_1 --gen inhibitory@real-pw8=%CTRLH%\inhibitory\poisson\seed_1 ^
  --out %ANA%\2026-11-19_conditional-report-v4-human

:end
echo ===== chain AG done ===== %date% %time%
goto :eof

:realctrl
rem child process: real H01 train spines of each binary type through the d8_pw8 chain (poisson control), local cache, no NAS
for %%t in (excitatory inhibitory) do (
  python scripts\neuron_model\make_real_controls.py --manifest %HMAN% --splits %SPLITH% --split train --metadata-root datasets ^
    --where cell_type_binary=%%t --controls poisson --poisson-config %CFG%\d8_pw8.yaml --n-samples 500 --seeds 1 --workers 8 --out %CTRLH%\%%t
)
echo done > %CTRLH%\_done.txt
goto :eof
