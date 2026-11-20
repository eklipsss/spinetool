@echo off
rem 2026-11-21, chain AI (predefense 26.11). GPU ~3.5 h, then CPU evaluation ~5-6 h; ~9 h in total.
rem Run after git pull, independent of the other chains (AH must be finished: it made the v4 g1.5 type sets and the val floor).
rem
rem AI1 (GPU) MoGen v4, mouse 23P and 6P-CT at guidance 3 (300 each, seed 1, d8_pw8). AH4c: v4 at guidance 1.5 reproduced 0 of 9 real
rem           23P / 6P-CT differences. Does the generated difference grow with the guidance? grows = the model sees the detailed type,
rem           but weakly; does not grow = the detailed type is not used. Pair 23P / 6P-CT: the largest real difference (9 of 11 metrics).
rem AI2 (CPU, background, after AI1) conditional report on 23P / 6P-CT: v4 g1.5 (AH sets), v4 g3, real-pw8val floor (AH2)
rem           -> 2026-11-21_conditional-report-v4-type-dose                                             log logs_AI_dose.txt
rem AI3 (GPU) second generation seed for MoGen A and v3 (unconditional, 500 each, checkpoint latest as seed 1, d8_pw8):
rem           the A vs v3 conclusion of AF rests on ONE generation seed. New sets <model dir>\seed_2 next to the seed_1 of AF.
rem AI4 (CPU) real controls of the same neurons (splits_v1_val_minus_dev, part test) with seeds 1 and 2 (300 each, independent subsets)
rem AI5 (CPU) evaluate_generators.py, v3 (500) vs A (500) vs real_poisson (300), generation seeds 1 and 2, the same protocol as AF
rem           (morphometric memorization on, Chamfer memorization off) + classifier ablation + tails
rem           -> <date>_generator-evaluation-v3-vs-A-n500-2seeds
rem Log: scripts\neuron_model\overnight\2026-11-21_AI_seed2-and-type-dose.bat > logs_AI.txt 2>&1

set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set SPLITS=data\processed\neuron-model\splits
set SPLITH=%SPLITS%\splits_v1_h01.parquet
set RUNS=runs\training\neuron-model
set ANA=runs\analysis\neuron-model
set CFG=configs\neuron-model\reconstruction\poisson_sweep
set V3=%RUNS%\2026-11-03_015237_mogen-conditional-v3
set V4=%RUNS%\2026-11-17_085545_mogen-conditional-v4-h01
set MA=%RUNS%\2026-10-20_213103_mogen-ft-dev
set P3=%V3%\meshes_poisson_d8_pw8
set P4=%V4%\meshes_poisson_d8_pw8
set MAN=%MA%\meshes_poisson_d8_pw8_n500
set CTRL=%ANA%\2026-11-20_real-controls-val\d8_pw8
set CMP2=%ANA%\2026-11-21_v3-vs-A-seeds
set FULL=C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/manifest_minnie65_full.parquet
set MORPH=C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/morphometrics_minnie65_full.parquet
set MARK=%ANA%\2026-11-21_AI_markers

cd /d %REPO%
if "%1"=="dose" goto :dose
if not exist %V4% (echo NO v4 RUN & goto :end)
if not exist %P4%\v4_mouse_23P_g1.5_seed_1 (echo NO v4 g1.5 type sets of chain AH & goto :end)
if not exist %CTRL%\23P\poisson\seed_1 (echo NO val floor of chain AH & goto :end)
if not exist %MAN%\seed_1 (echo NO A n500 seed 1 set of chain AF & goto :end)
if not exist %P3%\seed_1 (echo NO v3 seed 1 set & goto :end)
if not exist %SPLITS%\splits_v1_val_minus_dev.parquet (echo NO splits_v1_val_minus_dev & goto :end)
if not exist %MARK% mkdir %MARK%
if exist %MARK%\dose_done.txt del %MARK%\dose_done.txt

echo ===== [AI1] MoGen v4: mouse 23P and 6P-CT at guidance 3 (GPU) ===== %date% %time%
for %%c in (23P 6P-CT) do (
  if exist %P4%\v4_mouse_%%c_g3_seed_1\sample_00299 (echo set v4_mouse_%%c_g3 already reconstructed, skipped) else (
    python scripts\neuron_model\generate_mogen.py --run %V4% --checkpoint latest --condition species=mouse,cell_type=%%c --guidance 3 --n-samples 300 --seed 1 --tag v4_mouse_%%c_g3_seed_1 --no-mesh
    if errorlevel 1 (echo GENERATION %%c FAILED) else (
      python scripts\neuron_model\reconstruct_mogen_poisson.py --run %V4% --set v4_mouse_%%c_g3_seed_1 --out %P4% --poisson-config %CFG%\d8_pw8.yaml --workers 8
    )
  )
)

echo ===== [AI2] dose report in the background (logs_AI_dose.txt) ===== %date% %time%
start "AI_dose" /b cmd /c ""%~f0" dose > logs_AI_dose.txt 2>&1"

echo ===== [AI3] second generation seed: v3 and A, 500 each, unconditional (GPU) ===== %date% %time%
echo checkpoint of the existing sets (must be :latest:):
for /f "delims=" %%F in ('dir /s /b %MA%\meshes\seed_1_n500\*.json 2^>nul') do (findstr /i "checkpoint_id" "%%F" & goto :ckA)
:ckA
for /f "delims=" %%F in ('dir /s /b %V3%\meshes\seed_1\*.json 2^>nul') do (findstr /i "checkpoint_id" "%%F" & goto :ckV3)
:ckV3
if exist %P3%\seed_2 (echo %P3%\seed_2 exists, v3 generation skipped - check that it has 500 samples) else (
  python scripts\neuron_model\generate_mogen.py --run %V3% --checkpoint latest --n-samples 500 --seed 2 --tag seed_2_n500 --no-mesh
  if errorlevel 1 (echo v3 GENERATION FAILED) else (
    python scripts\neuron_model\reconstruct_mogen_poisson.py --run %V3% --meshes-dir %V3%\meshes --set seed_2_n500 --out %P3% --poisson-config %CFG%\d8_pw8.yaml --workers 8
    ren %P3%\seed_2_n500 seed_2
  )
)
if exist %MAN%\seed_2 (echo %MAN%\seed_2 exists, A generation skipped - check that it has 500 samples) else (
  python scripts\neuron_model\generate_mogen.py --run %MA% --checkpoint latest --n-samples 500 --seed 2 --tag seed_2_n500 --no-mesh
  if errorlevel 1 (echo A GENERATION FAILED) else (
    python scripts\neuron_model\reconstruct_mogen_poisson.py --run %MA% --meshes-dir %MA%\meshes --set seed_2_n500 --out %MAN% --poisson-config %CFG%\d8_pw8.yaml --workers 8
    ren %MAN%\seed_2_n500 seed_2
  )
)
if not exist %P3%\seed_2 (echo NO v3 seed_2 - evaluation skipped & goto :waitdose)
if not exist %MAN%\seed_2 (echo NO A seed_2 - evaluation skipped & goto :waitdose)

echo ===== [AI4] real controls, seeds 1 and 2 (300 each) ===== %date% %time%
if exist %CMP2%\real\poisson\seed_2\sample_00299 (echo real controls exist, skipped) else (
  python scripts\neuron_model\make_real_controls.py --manifest cache\minnie65_v1_60k\manifest.parquet --splits %SPLITS%\splits_v1_val_minus_dev.parquet --split test ^
    --controls poisson --poisson-config %CFG%\d8_pw8.yaml --n-samples 300 --seeds 1 2 --workers 8 --out %CMP2%\real
)

echo ===== [AI5] evaluation v3 (500) vs A (500) vs real controls (300), generation seeds 1 and 2 ===== %date% %time%
python scripts\neuron_model\evaluate_generators.py ^
  --model mogen_v3=%P3% --model mogen_A=%MAN% --model real_poisson=%CMP2%\real\poisson ^
  "data.manifests=['%FULL%']" data.path_mode=absolute data.splits=%SPLITS%\splits_v1_val_minus_dev.parquet ^
  data.bbox=%SPLITS%\bbox_splits_v1.json data.drop_unassigned=true "protocol.generation_seeds=[1,2]" real.morphometrics_table=%MORPH% ^
  real.chamfer_memorization=false name=generator-evaluation-v3-vs-A-n500-2seeds
if errorlevel 1 (echo EVALUATION FAILED & goto :waitdose)
for /f "delims=" %%d in ('dir /b /ad /o-n %ANA%\*_generator-evaluation-v3-vs-A-n500-2seeds') do (set EVAL=%ANA%\%%d& goto :evalfound)
:evalfound
echo evaluation run: %EVAL%
python scripts\neuron_model\classifier_ablation.py --eval-run %EVAL% --reference real_poisson --models mogen_v3 mogen_A
python scripts\neuron_model\morphometric_tails.py --eval-run %EVAL% --metrics Volume Area Length ConvexHullVolume
type %EVAL%\postprocessed\summary.md

:waitdose
echo ===== waiting for the dose report ===== %date% %time%
:waitloop
if exist %MARK%\dose_done.txt goto :end
ping -n 61 127.0.0.1 >nul
goto :waitloop

:end
echo ===== chain AI done ===== %date% %time%
goto :eof

:dose
rem child process: 23P / 6P-CT, v4 at guidance 1.5 and 3 against the real val spines, real-pw8val floor
python scripts\neuron_model\conditional_generation_report.py --block cell_type --datasets minnie65 --morphometrics-table %MORPH% ^
  --splits %SPLITH% --split val ^
  --gen 23P@v4g1.5-pw8=%P4%\v4_mouse_23P_g1.5_seed_1 --gen 6P-CT@v4g1.5-pw8=%P4%\v4_mouse_6P-CT_g1.5_seed_1 ^
  --gen 23P@v4g3-pw8=%P4%\v4_mouse_23P_g3_seed_1 --gen 6P-CT@v4g3-pw8=%P4%\v4_mouse_6P-CT_g3_seed_1 ^
  --gen 23P@real-pw8val=%CTRL%\23P\poisson\seed_1 --gen 6P-CT@real-pw8val=%CTRL%\6P-CT\poisson\seed_1 ^
  --out %ANA%\2026-11-21_conditional-report-v4-type-dose
echo done > %MARK%\dose_done.txt
goto :eof
