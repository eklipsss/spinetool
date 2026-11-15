@echo off
rem 2026-11-16, chain Z (CPU only, ~4-5 h). Run after git pull. Repeats the steps of chain Y that failed:
rem Z1 (= Y6) real TRAIN spines of each type through the old chain (p0_current) and d8_pw8. In Y every job failed with
rem    FileNotFoundError on \\LabidNAS\...\pointclouds.npz (the NAS was unreachable that night). Now the point clouds are read
rem    from the LOCAL cache of splits_v1 (cache\minnie65_v1_60k: 60 000 random train spines, point clouds 8192 variants 1-2;
rem    the Poisson control needs only the cloud, variant 1). New folder; the failed Y folder is not reused.
rem Z2 / Z3 (= Y7 / Y8) conditional reports. MC is left out of the detailed types: splits_v1 has no MC neuron in val.
rem Z4 (= Y9) section 4 of the global report.
rem Z5 (= Y11) reassembly of the two Y sets: the slivers are now removed once more AFTER the mesh simplification
rem    (in Y 1153 of 1154 extra components were debris < 10 000 nm^2 pinched off by the 5 nm simplification). New folder assembly_z.
rem Log: scripts\neuron_model\overnight\2026-11-16_Z_rerun-controls-assembly.bat > logs_Z.txt 2>&1

set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set B=C:\Users\Student\LipsEkaterina\data\minnie65\preprocessed_branches
set SPLIT=data\processed\neuron-model\splits\splits_v1.parquet
set RUNS=runs\training\neuron-model
set ANA=runs\analysis\neuron-model
set TPV=%ANA%\tp_generation\sigma_by_type
set TPS=%ANA%\tp_generation\subtype_bending
set NEWGEN=%TPS%\generated\2026-11-15_045430_generated_seed1
set NEWREAL=%TPS%\real_t\2026-11-15_045535_real_t_seed1
set GT=%ANA%\2026-11-06_013458_global-tables
set V3=%RUNS%\2026-11-03_015237_mogen-conditional-v3
set PW8=%V3%\meshes_poisson_d8_pw8
set MOGA=%RUNS%\2026-10-20_213103_mogen-ft-dev\meshes\seed_1
set CFG=configs\neuron-model\reconstruction\poisson_sweep
set CTRL=%ANA%\2026-11-16_real-controls-by-type
set MANIFEST=cache\minnie65_v1_60k\manifest.parquet
set MORPH=C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/morphometrics_minnie65_full.parquet
set BIN_EXC=cond_cell_type_binary-excitatory_g1.25_seed_1_n500
set BIN_INH=cond_cell_type_binary-inhibitory_g1.25_seed_1_n500

cd /d %REPO%
if not exist %MANIFEST% (echo MISSING %MANIFEST% & goto :end)
for /f "delims=" %%d in ('dir /b /ad /o-n %TPV%\generated') do (set OLDGEN=%TPV%\generated\%%d& goto :oldfound)
:oldfound
echo old T+P set (chain V): %OLDGEN%

echo ===== [Z1] real TRAIN spines of each type through the old chain and d8_pw8 (local cache) ===== %date% %time%
for %%c in (p0_current d8_pw8) do (
  for %%t in (excitatory inhibitory) do (
    python scripts\neuron_model\make_real_controls.py --manifest %MANIFEST% --splits %SPLIT% --split train --metadata-root datasets ^
      --where cell_type_binary=%%t --controls poisson --poisson-config %CFG%\%%c.yaml --n-samples 500 --seeds 1 --workers 8 --out %CTRL%\%%c\%%t
  )
  for %%t in (4P 6P-CT BPC) do (
    python scripts\neuron_model\make_real_controls.py --manifest %MANIFEST% --splits %SPLIT% --split train --metadata-root datasets ^
      --where cell_type=%%t --controls poisson --poisson-config %CFG%\%%c.yaml --n-samples 200 --seeds 1 --workers 8 --out %CTRL%\%%c\%%t
  )
)

echo ===== [Z2] conditional report, binary type ===== %date% %time%
set ZBIN=%ANA%\2026-11-16_conditional-report-pw8-binary
python scripts\neuron_model\conditional_generation_report.py --block cell_type_binary --morphometrics-table %MORPH% --splits %SPLIT% --split val ^
  --gen excitatory@g1.25-pw0=%V3%\meshes\%BIN_EXC% --gen inhibitory@g1.25-pw0=%V3%\meshes\%BIN_INH% ^
  --gen excitatory@g1.25-pw8=%PW8%\%BIN_EXC% --gen inhibitory@g1.25-pw8=%PW8%\%BIN_INH% ^
  --gen excitatory@real-pw0=%CTRL%\p0_current\excitatory\poisson\seed_1 --gen inhibitory@real-pw0=%CTRL%\p0_current\inhibitory\poisson\seed_1 ^
  --gen excitatory@real-pw8=%CTRL%\d8_pw8\excitatory\poisson\seed_1 --gen inhibitory@real-pw8=%CTRL%\d8_pw8\inhibitory\poisson\seed_1 ^
  --out %ZBIN%

echo ===== [Z3] conditional report, detailed types ===== %date% %time%
set ZTYPE=%ANA%\2026-11-16_conditional-report-pw8-types
set GENS=
for %%t in (4P 6P-CT BPC) do call :addtype %%t
python scripts\neuron_model\conditional_generation_report.py --block cell_type --morphometrics-table %MORPH% --splits %SPLIT% --split val %GENS% --out %ZTYPE%
goto :z4

:addtype
set GENS=%GENS% --gen %1@g1.5-pw0=%V3%\meshes\cond_cell_type-%1_g1.5_seed_1 --gen %1@g1.5-pw8=%PW8%\cond_cell_type-%1_g1.5_seed_1
set GENS=%GENS% --gen %1@real-pw0=%CTRL%\p0_current\%1\poisson\seed_1 --gen %1@real-pw8=%CTRL%\d8_pw8\%1\poisson\seed_1
goto :eof

:z4
echo ===== [Z4] section 4 of the global report ===== %date% %time%
if not exist %ZBIN%\results.json (echo no binary report: section 4 skipped & goto :z5)
rem Y9 failed after copying the X1 tables to section4_X1 (the tables in %GT% are still those of X1): copy only if missing
if not exist %GT%\section4_X1 (mkdir %GT%\section4_X1 & copy /y %GT%\generated_*.* %GT%\section4_X1\ >nul)
set SPINES=--generated-spines %ZBIN%@MoGen-v3
if exist %ZTYPE%\results.json set SPINES=%SPINES% --generated-spines %ZTYPE%@MoGen-v3-types
python scripts\neuron_model\build_global_tables.py --add-to %GT% --generated-tp %OLDGEN%@T0+P1b2 --generated-tp %NEWGEN%@T0bend+P1b2sub %SPINES% ^
  --generated-reference minnie65 --generated-reference-splits %SPLIT% --generated-reference-split val
python scripts\neuron_model\build_global_report.py --dir %GT%

:z5
echo ===== [Z5] reassembly of the Y sets (slivers removed after simplification) ===== %date% %time%
for %%s in (%NEWGEN% %NEWREAL%) do (
  echo ----- assembly of %%s -----
  python scripts\neuron_model\assemble_branches.py --tp-dir %%s --spines %MOGA% ^
    --spines-by-type excitatory=%PW8%\%BIN_EXC% inhibitory=%PW8%\%BIN_INH% ^
    --context-spines %B%\branch_spines.parquet --n-branches 300 --seed 1 --out %%s\assembly_z
  python scripts\neuron_model\evaluate_assembled_branches.py --tp-dir %%s --assembly %%s\assembly_z --out %%s\assembly_z\evaluation
  python scripts\neuron_model\view_assembled_branches.py --assembly %%s\assembly_z --max 6
)

:end
echo ===== Z done ===== %date% %time%
