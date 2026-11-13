@echo off
rem 2026-11-14, chain X (CPU only, ~3-5 h; X3 is the long one). Run after git pull.
rem X0 (minutes) re-evaluation with thinning of the sets evaluated before 2026-11-11 (step V0 of chain V that did not run:
rem    the log of V starts at V1) + the newest sigma_by_type sets (V5) -> NEW folders evaluation_thinned*, old ones stay.
rem X1 (minutes) section 4 of the global report (generated dendrites and spines against the real ones) into the
rem    global-tables folder of 2026-11-06; reference spines: val neurons of splits_v1 (the test stays untouched).
rem X2 (~1.5-2.5 h) cross-validation of P0 and T0 by NEURON on train + val of splits_v1 (cv_tp_neurons.py, 5 folds,
rem    stratified by cell_type_binary; P0 x 3 repetitions, 2 variants; T0 parabola_mix x 1 repetition): are the 8 val
rem    inhibitory neurons poorer under any split (property of the neurons) or only an accident of the split?
rem X3 (~1-3 h) whole branches T + P + S: 300 branches of the newest generated set (T0 + P1) and 300 of the real-trunk set,
rem    spines from the conditional MoGen v3 by the cell type of the branch (500 spines per type, w = 1.25), default pool
rem    MoGen A; the pool spine is chosen by the trunk radius (--context-spines); then the evaluation of the assembled
rem    branches and an HTML viewer of 6 branches.
rem Log: scripts\neuron_model\overnight\2026-11-14_X_cv-and-assembly.bat > logs_X.txt 2>&1

set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set B=C:\Users\Student\LipsEkaterina\data\minnie65\preprocessed_branches
set SPLIT=data\processed\neuron-model\splits\splits_v1.parquet
set RUNS=runs\training\neuron-model
set ANA=runs\analysis\neuron-model
set TT=%ANA%\tp_generation
set TPV=%TT%\sigma_by_type
set GT=%ANA%\2026-11-06_013458_global-tables
set COND=%ANA%\2026-11-04_094722_conditional-report
set V3=%RUNS%\2026-11-03_015237_mogen-conditional-v3\meshes
set MOGA=%RUNS%\2026-10-20_213103_mogen-ft-dev\meshes\seed_1

cd /d %REPO%
for /f "delims=" %%d in ('dir /b /ad /o-n %TPV%\generated') do (set GEN=%TPV%\generated\%%d& goto :genfound)
:genfound
for /f "delims=" %%d in ('dir /b /ad /o-n %TPV%\real_t') do (set REALT=%TPV%\real_t\%%d& goto :realfound)
:realfound
echo generated set: %GEN%
echo real-trunk set: %REALT%

echo ===== [X0] re-evaluation with thinning ===== %date% %time%
for %%s in (%TT%\inhibitory\generated\2026-11-10_114858_generated_seed1 %TT%\inhibitory\real_t\2026-11-10_115009_real_t_seed1 %TT%\counts_real\p1b2\2026-11-10_174333_real_t_counts-real_seed1 %GEN% %REALT%) do (
  if exist %%s\trunks.npz (
    echo ----- %%s -----
    python scripts\neuron_model\evaluate_assembled_branches.py --tp-dir %%s --out %%s\evaluation_thinned
    python scripts\neuron_model\evaluate_assembled_branches.py --tp-dir %%s --cell-type-binary excitatory --out %%s\evaluation_thinned_excitatory
    python scripts\neuron_model\evaluate_assembled_branches.py --tp-dir %%s --cell-type-binary inhibitory --out %%s\evaluation_thinned_inhibitory
  ) else (
    echo MISSING %%s
  )
)

echo ===== [X1] global report, section 4 ===== %date% %time%
if not exist %GT%\run_info.json (echo MISSING %GT% & goto :x2)
if exist %COND%\generated_metrics.parquet (
  python scripts\neuron_model\build_global_tables.py --add-to %GT% --generated-tp %GEN%@T0+P1b2 --generated-spines %COND%@MoGen-v3 ^
    --generated-reference minnie65 --generated-reference-splits %SPLIT% --generated-reference-split val
) else (
  echo no generated_metrics.parquet in %COND%: section 4 for T and P only
  python scripts\neuron_model\build_global_tables.py --add-to %GT% --generated-tp %GEN%@T0+P1b2
)
python scripts\neuron_model\build_global_report.py --dir %GT%

:x2
echo ===== [X2] cross-validation of P0 and T0 by neuron ===== %date% %time%
python scripts\neuron_model\cv_tp_neurons.py --branches-root %B% --splits %SPLIT% --k 5 --repeats 3 --models p0 t0

echo ===== [X3] whole branches T + P + S ===== %date% %time%
if not exist %V3%\cond_cell_type_binary-excitatory_g1.25_seed_1_n500 (echo MISSING the MoGen v3 spine sets & goto :end)
for %%s in (%GEN% %REALT%) do (
  echo ----- assembly of %%s -----
  python scripts\neuron_model\assemble_branches.py --tp-dir %%s --spines %MOGA% ^
    --spines-by-type excitatory=%V3%\cond_cell_type_binary-excitatory_g1.25_seed_1_n500 inhibitory=%V3%\cond_cell_type_binary-inhibitory_g1.25_seed_1_n500 ^
    --context-spines %B%\branch_spines.parquet --n-branches 300 --seed 1 --out %%s\assembly_x
  python scripts\neuron_model\evaluate_assembled_branches.py --tp-dir %%s --assembly %%s\assembly_x --out %%s\assembly_x\evaluation
  python scripts\neuron_model\view_assembled_branches.py --assembly %%s\assembly_x --max 6
)

:end
echo ===== X done ===== %date% %time%
