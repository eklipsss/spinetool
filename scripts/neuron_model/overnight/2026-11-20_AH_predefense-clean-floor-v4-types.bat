@echo off
rem 2026-11-20, chain AH (pre-defence night 1; GPU ~3 h, CPU in parallel, reports ~3 h in parallel; ~7-9 h in total).
rem Run after git pull. Replaces chain AE (never run): AE1-AE2 are here as AH1, AE3 (v4 species sets) was already done by AG, AE4 dropped
rem (the conditional reports measure the size bias against the real spines of the same type).
rem
rem Why: in every conditional report so far the floor "real spines through the d8_pw8 chain" was made from TRAIN neurons, while the real
rem side of the comparison is the VAL neurons -> the floor mixed the chain effect with a train/val neuron difference (strong for the 8 val
rem inhibitory neurons). Here the floor is made from the SAME val neurons (real-pw8val): then "real-pw8val vs real" measures the chain only.
rem
rem AH1 (CPU, background child "tp")  T0 with the radius-dependent wiggle scale on Minnie65 + H01, T/P generation + evaluation per dataset on
rem                                   val, neuron CV of T0 on the two datasets                                   -> logs_AH_tp.txt
rem AH2 (CPU, background child "ctrl") real val spines through d8_pw8: Minnie65 excitatory / inhibitory / 4P / 6P-CT / 23P / 6P-IT (meshes'
rem                                   point clouds from the NAS, full manifest), H01 excitatory / inhibitory (local cache h01_v1, spines
rem                                   without a neck already excluded)                                          -> logs_AH_ctrl.txt
rem AH3 (GPU, foreground)             MoGen v4, mouse x detailed type 4P / 6P-CT / 23P / 6P-IT, guidance 1.5 (as the v3 type sets), 300 each,
rem                                   d8_pw8 Poisson chain
rem AH4 (CPU, after AH2 and AH3, three reports in parallel + species table)
rem     a  mouse, binary type: v4, v3, real-pw8val                                      -> 2026-11-20_conditional-report-v4-mouse-valfloor
rem     b  human, binary type: v4, real-pw8val (only 3 val inhibitory neurons in H01:
rem        the inhibitory pool is skipped by the report, read as descriptive)           -> 2026-11-20_conditional-report-v4-human-valfloor
rem     c  mouse, detailed type: v4 (4 types), v3 (4P, 6P-CT), real-pw8val              -> 2026-11-20_conditional-report-v4-types-valfloor
rem     d  species table (ratios of medians) with the val floor                         -> 2026-11-20_species-generation-v4-valfloor
rem Log: scripts\neuron_model\overnight\2026-11-20_AH_predefense-clean-floor-v4-types.bat > logs_AH.txt 2>&1

set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set B=C:\Users\Student\LipsEkaterina\data\minnie65\preprocessed_branches
set H01B=C:\Users\Student\LipsEkaterina\data\h01\preprocessed_branches
set ROOTS=%B%;%H01B%
set SPLITS=data\processed\neuron-model\splits
set SPLITH=%SPLITS%\splits_v1_h01.parquet
set DETACHED=%SPLITS%\h01_detached_spines.parquet
set RUNS=runs\training\neuron-model
set ANA=runs\analysis\neuron-model
set CFG=configs\neuron-model\reconstruction\poisson_sweep
set P1H=%RUNS%\2026-11-17_072620_p1
set TPH=%ANA%\tp_generation\minnie_h01_acfit
set V3=%RUNS%\2026-11-03_015237_mogen-conditional-v3
set V4=%RUNS%\2026-11-17_085545_mogen-conditional-v4-h01
set P3=%V3%\meshes_poisson_d8_pw8
set P4=%V4%\meshes_poisson_d8_pw8
set FULLM=C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/manifest_minnie65_full.parquet
set HMAN=cache\h01_v1\manifest.parquet
set MORPHM=C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/morphometrics_minnie65_full.parquet
set MORPHH=C:/Users/Student/LipsEkaterina/data/h01/preprocessed_v2/H01_Seg/morphometrics.parquet
set CTRL=%ANA%\2026-11-20_real-controls-val\d8_pw8
set CTRLH=%ANA%\2026-11-20_real-controls-val-h01\d8_pw8
set MARK=%ANA%\2026-11-20_AH_markers
set V3E=cond_cell_type_binary-excitatory_g1.25_seed_1_n500
set V3I=cond_cell_type_binary-inhibitory_g1.25_seed_1_n500

cd /d %REPO%
if "%1"=="tp" goto :tp
if "%1"=="ctrl" goto :ctrl
if "%1"=="repmouse" goto :repmouse
if "%1"=="rephuman" goto :rephuman

if not exist %V4% (echo NO v4 RUN & goto :end)
if not exist %P1H% (echo NO P1 RUN %P1H% & goto :end)
if not exist %FULLM% (echo NO full Minnie65 manifest %FULLM% & goto :end)
if not exist %HMAN% (echo NO H01 CACHE %HMAN% & goto :end)
if not exist %P4%\v4_mouse_excitatory_g1.25_seed_1 (echo NO v4 species sets of chain AG & goto :end)
if not exist %MARK% mkdir %MARK%
del /q %MARK%\*.txt 2>nul

echo ===== [AH1] T0 + T/P generation + CV in the background (logs_AH_tp.txt) ===== %date% %time%
start "AH_tp" /b cmd /c ""%~f0" tp > logs_AH_tp.txt 2>&1"
echo ===== [AH2] val real controls in the background (logs_AH_ctrl.txt) ===== %date% %time%
start "AH_ctrl" /b cmd /c ""%~f0" ctrl > logs_AH_ctrl.txt 2>&1"

echo ===== [AH3] MoGen v4: mouse x detailed type, 300 each, guidance 1.5 (GPU) ===== %date% %time%
for %%c in (4P 6P-CT 23P 6P-IT) do (
  if exist %P4%\v4_mouse_%%c_g1.5_seed_1\sample_00299 (echo set v4_mouse_%%c already reconstructed, skipped) else (
    python scripts\neuron_model\generate_mogen.py --run %V4% --checkpoint latest --condition species=mouse,cell_type=%%c --guidance 1.5 --n-samples 300 --seed 1 --tag v4_mouse_%%c_g1.5_seed_1 --no-mesh
    if errorlevel 1 (echo GENERATION %%c FAILED) else (
      python scripts\neuron_model\reconstruct_mogen_poisson.py --run %V4% --set v4_mouse_%%c_g1.5_seed_1 --out %P4% --poisson-config %CFG%\d8_pw8.yaml --workers 8
    )
  )
)
echo GPU part done %date% %time%

echo ===== waiting for the val real controls ===== %date% %time%
:waitctrl
if exist %MARK%\ctrl_done.txt goto :ctrldone
ping -n 61 127.0.0.1 >nul
goto :waitctrl
:ctrldone
echo val controls done %date% %time%

echo ===== [AH4a, AH4b] binary-type reports in the background ===== %date% %time%
start "AH_repmouse" /b cmd /c ""%~f0" repmouse > logs_AH_repmouse.txt 2>&1"
start "AH_rephuman" /b cmd /c ""%~f0" rephuman > logs_AH_rephuman.txt 2>&1"

echo ===== [AH4c] conditional report, MOUSE, detailed type ===== %date% %time%
python scripts\neuron_model\conditional_generation_report.py --block cell_type --datasets minnie65 --morphometrics-table %MORPHM% ^
  --splits %SPLITH% --split val ^
  --gen 4P@v4-pw8=%P4%\v4_mouse_4P_g1.5_seed_1 --gen 6P-CT@v4-pw8=%P4%\v4_mouse_6P-CT_g1.5_seed_1 ^
  --gen 23P@v4-pw8=%P4%\v4_mouse_23P_g1.5_seed_1 --gen 6P-IT@v4-pw8=%P4%\v4_mouse_6P-IT_g1.5_seed_1 ^
  --gen 4P@v3-pw8=%P3%\cond_cell_type-4P_g1.5_seed_1 --gen 6P-CT@v3-pw8=%P3%\cond_cell_type-6P-CT_g1.5_seed_1 ^
  --gen 4P@real-pw8val=%CTRL%\4P\poisson\seed_1 --gen 6P-CT@real-pw8val=%CTRL%\6P-CT\poisson\seed_1 ^
  --gen 23P@real-pw8val=%CTRL%\23P\poisson\seed_1 --gen 6P-IT@real-pw8val=%CTRL%\6P-IT\poisson\seed_1 ^
  --out %ANA%\2026-11-20_conditional-report-v4-types-valfloor

echo ===== [AH4d] species table with the val floor ===== %date% %time%
python scripts\neuron_model\compare_species_generation.py ^
  --gen mouse/excitatory@v4-pw8=%P4%\v4_mouse_excitatory_g1.25_seed_1 --gen mouse/inhibitory@v4-pw8=%P4%\v4_mouse_inhibitory_g1.25_seed_1 ^
  --gen human/excitatory@v4-pw8=%P4%\v4_human_excitatory_g1.25_seed_1 --gen human/inhibitory@v4-pw8=%P4%\v4_human_inhibitory_g1.25_seed_1 ^
  --gen mouse/excitatory@real-pw8val=%CTRL%\excitatory\poisson\seed_1 --gen mouse/inhibitory@real-pw8val=%CTRL%\inhibitory\poisson\seed_1 ^
  --gen human/excitatory@real-pw8val=%CTRLH%\excitatory\poisson\seed_1 --gen human/inhibitory@real-pw8val=%CTRLH%\inhibitory\poisson\seed_1 ^
  --minnie-table %MORPHM% --h01-table %MORPHH% --h01-detached %DETACHED% --splits %SPLITH% --split val ^
  --out %ANA%\2026-11-20_species-generation-v4-valfloor

echo ===== waiting for the background jobs (tp, repmouse, rephuman) ===== %date% %time%
:waitall
if not exist %MARK%\tp_done.txt goto :sleepall
if not exist %MARK%\repmouse_done.txt goto :sleepall
if not exist %MARK%\rephuman_done.txt goto :sleepall
goto :end
:sleepall
ping -n 61 127.0.0.1 >nul
goto :waitall

:end
echo ===== chain AH done ===== %date% %time%
goto :eof

rem ---- background children ---------------------------------------------------------------------------------------
:tp
echo ===== [AH1] T0 (radius-dependent wiggle scale) on Minnie65 + H01 ===== %date% %time%
python scripts\neuron_model\fit_t0.py --config configs\neuron-model\trunk\t0.yaml "data.branches_root=%ROOTS%" data.splits=%SPLITH% "model.mean_models=[parabola_mix]"
if errorlevel 1 (echo T0 FAILED & goto :tpend)
for /f "delims=" %%d in ('dir /b /ad /o-n %RUNS%\*_t0') do (set T0H=%RUNS%\%%d& goto :t0found)
:t0found
echo new T0 run: %T0H%
python scripts\neuron_model\diagnose_t0_direction.py --t0-run %T0H% "data.branches_root=%ROOTS%" data.splits=%SPLITH%
for %%k in (minnie65 h01) do (
  python scripts\neuron_model\generate_tp.py --t0-run %T0H% --p0-run %P1H% --mode generated --n 1000 --seed 1 --split val --dataset %%k --out-root %TPH%\%%k\generated
  python scripts\neuron_model\generate_tp.py --t0-run %T0H% --p0-run %P1H% --mode real_t --n 1000 --seed 1 --split val --dataset %%k --out-root %TPH%\%%k\real_t
)
for %%k in (minnie65 h01) do (
  for %%m in (generated real_t) do call :evalnewest %TPH%\%%k\%%m
)
python scripts\neuron_model\cv_tp_neurons.py --branches-root "%ROOTS%" --splits %SPLITH% --k 5 --repeats 1 --models t0
:tpend
echo done > %MARK%\tp_done.txt
echo ===== AH1 done ===== %date% %time%
goto :eof

:evalnewest
for /f "delims=" %%d in ('dir /b /ad /o-n %1') do (
  python scripts\neuron_model\evaluate_assembled_branches.py --tp-dir %1\%%d
  goto :eof
)
goto :eof

:ctrl
rem real VAL spines through d8_pw8. Minnie65: full manifest (absolute NAS paths of the point clouds); H01: local cache h01_v1.
for %%t in (excitatory inhibitory) do (
  python scripts\neuron_model\make_real_controls.py --manifest %FULLM% --path-mode absolute --splits %SPLITH% --split val --metadata-root datasets ^
    --where cell_type_binary=%%t --controls poisson --poisson-config %CFG%\d8_pw8.yaml --n-samples 500 --seeds 1 --allow-fewer --workers 8 --out %CTRL%\%%t
)
for %%t in (4P 6P-CT 23P 6P-IT) do (
  python scripts\neuron_model\make_real_controls.py --manifest %FULLM% --path-mode absolute --splits %SPLITH% --split val --metadata-root datasets ^
    --where cell_type=%%t --controls poisson --poisson-config %CFG%\d8_pw8.yaml --n-samples 500 --seeds 1 --allow-fewer --workers 8 --out %CTRL%\%%t
)
for %%t in (excitatory inhibitory) do (
  python scripts\neuron_model\make_real_controls.py --manifest %HMAN% --splits %SPLITH% --split val --metadata-root datasets ^
    --where cell_type_binary=%%t --controls poisson --poisson-config %CFG%\d8_pw8.yaml --n-samples 500 --seeds 1 --allow-fewer --workers 8 --out %CTRLH%\%%t
)
echo done > %MARK%\ctrl_done.txt
echo ===== AH2 done ===== %date% %time%
goto :eof

:repmouse
python scripts\neuron_model\conditional_generation_report.py --block cell_type_binary --datasets minnie65 --morphometrics-table %MORPHM% ^
  --splits %SPLITH% --split val ^
  --gen excitatory@v4-pw8=%P4%\v4_mouse_excitatory_g1.25_seed_1 --gen inhibitory@v4-pw8=%P4%\v4_mouse_inhibitory_g1.25_seed_1 ^
  --gen excitatory@v3-pw8=%P3%\%V3E% --gen inhibitory@v3-pw8=%P3%\%V3I% ^
  --gen excitatory@real-pw8val=%CTRL%\excitatory\poisson\seed_1 --gen inhibitory@real-pw8val=%CTRL%\inhibitory\poisson\seed_1 ^
  --out %ANA%\2026-11-20_conditional-report-v4-mouse-valfloor
echo done > %MARK%\repmouse_done.txt
goto :eof

:rephuman
python scripts\neuron_model\conditional_generation_report.py --block cell_type_binary --datasets h01 --morphometrics-table %MORPHH% ^
  --exclude-keys %DETACHED% --splits %SPLITH% --split val ^
  --gen excitatory@v4-pw8=%P4%\v4_human_excitatory_g1.25_seed_1 --gen inhibitory@v4-pw8=%P4%\v4_human_inhibitory_g1.25_seed_1 ^
  --gen excitatory@real-pw8val=%CTRLH%\excitatory\poisson\seed_1 --gen inhibitory@real-pw8val=%CTRLH%\inhibitory\poisson\seed_1 ^
  --out %ANA%\2026-11-20_conditional-report-v4-human-valfloor
echo done > %MARK%\rephuman_done.txt
goto :eof
