@echo off
rem 2026-11-15, chain Y (CPU only, no GPU; ~7-9 h, overnight). Run after git pull.
rem
rem Part A, modules T and P (Y1-Y4, ~1.5 h):
rem Y1 T0 with the corrected bending fit (configs t0.yaml model.direction_options; direction_ms.py docstring): the curves of the fit
rem    are corrected by simulating the training branches (the linear curves were not the generated ones: on the chain-T fit the
rem    generated bending was 0.75-0.89 of real at 0.5-1 um and the long-lag tilt 1.08 -> tortuosity +0.037, kappa autocorrelation
rem    +0.05 in every CV fold), the tilt is fitted from 2 um, the bending amplitude (radius slope, compartment offsets) is
rem    estimated WITHIN neurons (chain T: pooled offsets apical -0.00 / basal +0.01, while the real apical branches of a neuron
rem    bend 21 % less than its basal ones, the generated 5 %). Only parabola_mix (chosen in chains T and V; the mean model is not changed).
rem Y2 P0 with the detailed cell type (covariates.py, count.py): one column per detailed type with >= 3 training neurons, a deviation
rem    from its binary type with the prior N(0, tau^2), tau in 0.1 / 0.25 / 0.5 / 1 by the val NLL, kept only if it helps; grid
rem    restricted to the variant chosen in chain V (zinb, length, neuron effect, own sigma for inhibitory). Calibration by type.
rem Y3 P1 (P1b2) on the new P0. Y4 generated set (new T0 + new P1) and real-trunk set, val of splits_v1, 1000 each, evaluation.
rem
rem Part B, spines (Y5-Y8, ~2.5-3 h): conditional MoGen v3 re-meshed with the chosen Poisson chain d8_pw8 + the chain floor:
rem Y5 the SAME v3 point clouds re-meshed with d8_pw8 (new folder meshes_poisson_d8_pw8; old meshes stay): binary type w = 1.25
rem    (500 each) and detailed types 4P / 6P-CT / BPC / MC w = 1.5 (200 each, the sets of the 2026-11-04 conditional report);
rem Y6 REAL spines of each type (train neurons of splits_v1; the test is not read) through the old chain (p0_current) and d8_pw8;
rem Y7 / Y8 conditional reports (binary / detailed types): g-pw0, g-pw8, real-pw0, real-pw8 against val of splits_v1.
rem    Reading: generator gap = (g-pw8 vs real) beyond (real-pw8 vs real).
rem
rem Part C (Y9-Y11, ~3-4 h):
rem Y9 section 4 of the global report into the 2026-11-06 global-tables folder (X1 tables copied to section4_X1 first): T+P old set
rem    (chain V) and new set (Y4) side by side, spines from Y7 / Y8.
rem Y10 neuron cross-validation (same folds as chain X2): P0 with detailed types (tau 0.5, primary: calibration by type) vs without,
rem    paired; T0 with the new bending fit on 5 folds (X2: kappa_rms, tortuosity, kappa autocorrelation significant in all 5).
rem Y11 whole branches T + P + S of the new sets with the sliver removal of the assembly (X3: 13-19 % invalid, mostly slivers).
rem Log: scripts\neuron_model\overnight\2026-11-15_Y_night-tp-spines.bat > logs_Y.txt 2>&1

set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set B=C:\Users\Student\LipsEkaterina\data\minnie65\preprocessed_branches
set SPLIT=data\processed\neuron-model\splits\splits_v1.parquet
set RUNS=runs\training\neuron-model
set ANA=runs\analysis\neuron-model
set TPV=%ANA%\tp_generation\sigma_by_type
set TPS=%ANA%\tp_generation\subtype_bending
set GT=%ANA%\2026-11-06_013458_global-tables
set V3=%RUNS%\2026-11-03_015237_mogen-conditional-v3
set PW8=%V3%\meshes_poisson_d8_pw8
set MOGA=%RUNS%\2026-10-20_213103_mogen-ft-dev\meshes\seed_1
set CFG=configs\neuron-model\reconstruction\poisson_sweep
set CTRL=%ANA%\2026-11-15_real-controls-by-type
set MANIFEST=cache\manifest_minnie65_meta.parquet
set MORPH=C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/morphometrics_minnie65_full.parquet
set BIN_EXC=cond_cell_type_binary-excitatory_g1.25_seed_1_n500
set BIN_INH=cond_cell_type_binary-inhibitory_g1.25_seed_1_n500
set NEWGEN=
set NEWREAL=

cd /d %REPO%
for /f "delims=" %%d in ('dir /b /ad /o-n %TPV%\generated') do (set OLDGEN=%TPV%\generated\%%d& goto :oldfound)
:oldfound
echo old T+P set (chain V): %OLDGEN%

rem ================================================================ part A
echo ===== [Y1] T0 with the corrected bending fit ===== %date% %time%
python scripts\neuron_model\fit_t0.py --config configs\neuron-model\trunk\t0.yaml data.branches_root=%B% data.splits=%SPLIT% "model.mean_models=[parabola_mix]"
if errorlevel 1 (echo T0 FAILED: parts A and C of T/P skipped & goto :partb)
for /f "delims=" %%d in ('dir /b /ad /o-n %RUNS%\*_t0') do (set T0NEW=%RUNS%\%%d& goto :t0found)
:t0found
echo new T0 run: %T0NEW%

echo ===== [Y2] P0 with the detailed cell type ===== %date% %time%
python scripts\neuron_model\fit_p0.py --config configs\neuron-model\placement\p0.yaml data.branches_root=%B% data.splits=%SPLIT% ^
  "model.families=[zinb]" "model.exposures=[length]" "model.random_effects=[true]" "model.sigma_by_cell_type=[true]"
if errorlevel 1 (echo P0 FAILED & goto :partb)
for /f "delims=" %%d in ('dir /b /ad /o-n %RUNS%\*_p0') do (set P0NEW=%RUNS%\%%d& goto :p0found)
:p0found
echo new P0 run: %P0NEW%
python scripts\neuron_model\calibrate_p0_counts.py --p0-run %P0NEW%

echo ===== [Y3] P1 on the new P0 ===== %date% %time%
python scripts\neuron_model\fit_p1.py --p0-run %P0NEW% --n-eval 400 --n-sim 19
if errorlevel 1 (echo P1 FAILED & goto :partb)
for /f "delims=" %%d in ('dir /b /ad /o-n %RUNS%\*_p1') do (set P1NEW=%RUNS%\%%d& goto :p1found)
:p1found
echo new P1 run: %P1NEW%

echo ===== [Y4] generated (new T0 + new P1) and real trunk, val, evaluation ===== %date% %time%
python scripts\neuron_model\generate_tp.py --t0-run %T0NEW% --p0-run %P1NEW% --mode generated --n 1000 --seed 1 --split val --out-root %TPS%\generated
python scripts\neuron_model\generate_tp.py --t0-run %T0NEW% --p0-run %P1NEW% --mode real_t --n 1000 --seed 1 --split val --out-root %TPS%\real_t
for /f "delims=" %%d in ('dir /b /ad /o-n %TPS%\generated') do (set NEWGEN=%TPS%\generated\%%d& goto :ng)
:ng
for /f "delims=" %%d in ('dir /b /ad /o-n %TPS%\real_t') do (set NEWREAL=%TPS%\real_t\%%d& goto :nr)
:nr
echo new sets: %NEWGEN% / %NEWREAL%
for %%s in (%NEWGEN% %NEWREAL%) do (
  echo ----- %%s: all / excitatory / inhibitory -----
  python scripts\neuron_model\evaluate_assembled_branches.py --tp-dir %%s
  python scripts\neuron_model\evaluate_assembled_branches.py --tp-dir %%s --cell-type-binary excitatory --out %%s\evaluation_excitatory
  python scripts\neuron_model\evaluate_assembled_branches.py --tp-dir %%s --cell-type-binary inhibitory --out %%s\evaluation_inhibitory
)

rem ================================================================ part B
:partb
echo ===== [Y5] v3 point clouds re-meshed with d8_pw8 ===== %date% %time%
for %%s in (%BIN_EXC% %BIN_INH% cond_cell_type-4P_g1.5_seed_1 cond_cell_type-6P-CT_g1.5_seed_1 cond_cell_type-BPC_g1.5_seed_1 cond_cell_type-MC_g1.5_seed_1) do (
  if exist %V3%\meshes\%%s (
    echo ----- %%s -----
    python scripts\neuron_model\reconstruct_mogen_poisson.py --run %V3% --meshes-dir %V3%\meshes --set %%s --out %PW8% --poisson-config %CFG%\d8_pw8.yaml --workers 8
  ) else (
    echo MISSING %V3%\meshes\%%s
  )
)

echo ===== [Y6] real TRAIN spines of each type through the old chain and d8_pw8 ===== %date% %time%
for %%c in (p0_current d8_pw8) do (
  for %%t in (excitatory inhibitory) do (
    python scripts\neuron_model\make_real_controls.py --manifest %MANIFEST% --path-mode absolute --splits %SPLIT% --split train --metadata-root datasets ^
      --where cell_type_binary=%%t --controls poisson --poisson-config %CFG%\%%c.yaml --n-samples 500 --seeds 1 --workers 8 --out %CTRL%\%%c\%%t
  )
  for %%t in (4P 6P-CT BPC MC) do (
    python scripts\neuron_model\make_real_controls.py --manifest %MANIFEST% --path-mode absolute --splits %SPLIT% --split train --metadata-root datasets ^
      --where cell_type=%%t --controls poisson --poisson-config %CFG%\%%c.yaml --n-samples 200 --seeds 1 --workers 8 --out %CTRL%\%%c\%%t
  )
)

echo ===== [Y7] conditional report, binary type ===== %date% %time%
set YBIN=%ANA%\2026-11-15_conditional-report-pw8-binary
python scripts\neuron_model\conditional_generation_report.py --block cell_type_binary --morphometrics-table %MORPH% --splits %SPLIT% --split val ^
  --gen excitatory@g1.25-pw0=%V3%\meshes\%BIN_EXC% --gen inhibitory@g1.25-pw0=%V3%\meshes\%BIN_INH% ^
  --gen excitatory@g1.25-pw8=%PW8%\%BIN_EXC% --gen inhibitory@g1.25-pw8=%PW8%\%BIN_INH% ^
  --gen excitatory@real-pw0=%CTRL%\p0_current\excitatory\poisson\seed_1 --gen inhibitory@real-pw0=%CTRL%\p0_current\inhibitory\poisson\seed_1 ^
  --gen excitatory@real-pw8=%CTRL%\d8_pw8\excitatory\poisson\seed_1 --gen inhibitory@real-pw8=%CTRL%\d8_pw8\inhibitory\poisson\seed_1 ^
  --out %YBIN%

echo ===== [Y8] conditional report, detailed types ===== %date% %time%
set YTYPE=%ANA%\2026-11-15_conditional-report-pw8-types
set GENS=
for %%t in (4P 6P-CT BPC MC) do call :addtype %%t
python scripts\neuron_model\conditional_generation_report.py --block cell_type --morphometrics-table %MORPH% --splits %SPLIT% --split val %GENS% --out %YTYPE%
goto :partc

:addtype
set GENS=%GENS% --gen %1@g1.5-pw0=%V3%\meshes\cond_cell_type-%1_g1.5_seed_1 --gen %1@g1.5-pw8=%PW8%\cond_cell_type-%1_g1.5_seed_1
set GENS=%GENS% --gen %1@real-pw0=%CTRL%\p0_current\%1\poisson\seed_1 --gen %1@real-pw8=%CTRL%\d8_pw8\%1\poisson\seed_1
goto :eof

rem ================================================================ part C
:partc
echo ===== [Y9] section 4 of the global report ===== %date% %time%
if not exist %GT%\section4_X1 mkdir %GT%\section4_X1
copy /y %GT%\generated_*.* %GT%\section4_X1\ >nul
set TPARGS=--generated-tp %OLDGEN%@T0+P1b2
if defined NEWGEN set TPARGS=%TPARGS% --generated-tp %NEWGEN%@T0bend+P1b2sub
python scripts\neuron_model\build_global_tables.py --add-to %GT% %TPARGS% ^
  --generated-spines %YBIN%@MoGen-v3 --generated-spines %YTYPE%@MoGen-v3-types ^
  --generated-reference minnie65 --generated-reference-splits %SPLIT% --generated-reference-split val
python scripts\neuron_model\build_global_report.py --dir %GT%

echo ===== [Y10] neuron cross-validation of the new P0 and T0 ===== %date% %time%
python scripts\neuron_model\cv_tp_neurons.py --branches-root %B% --splits %SPLIT% --k 5 --repeats 3 --models p0 t0 ^
  --p0-variants zinb:length:re:sg2:t0.5 zinb:length:re:sg2

echo ===== [Y11] whole branches T + P + S of the new sets (sliver removal) ===== %date% %time%
if not defined NEWGEN (echo no new T+P sets & goto :end)
set POOL=%PW8%
if not exist %PW8%\%BIN_EXC% set POOL=%V3%\meshes
if not exist %POOL%\%BIN_EXC% (echo MISSING the MoGen v3 spine sets & goto :end)
echo spine pools: %POOL%
for %%s in (%NEWGEN% %NEWREAL%) do (
  echo ----- assembly of %%s -----
  python scripts\neuron_model\assemble_branches.py --tp-dir %%s --spines %MOGA% ^
    --spines-by-type excitatory=%POOL%\%BIN_EXC% inhibitory=%POOL%\%BIN_INH% ^
    --context-spines %B%\branch_spines.parquet --n-branches 300 --seed 1 --out %%s\assembly_y
  python scripts\neuron_model\evaluate_assembled_branches.py --tp-dir %%s --assembly %%s\assembly_y --out %%s\assembly_y\evaluation
  python scripts\neuron_model\view_assembled_branches.py --assembly %%s\assembly_y --max 6
)

:end
echo ===== Y done ===== %date% %time%
