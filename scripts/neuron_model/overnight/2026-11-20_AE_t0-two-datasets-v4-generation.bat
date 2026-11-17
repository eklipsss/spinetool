@echo off
rem 2026-11-18, chain AE (CPU + GPU, ~4-6 h). Run after git pull. Needs the results of AB (T0/P0/P1 runs below) and of AC.
rem
rem AC result: the radius-dependent wiggle scale (gamma ~ 0.65) removed the kappa autocorrelation defect in the neuron CV
rem (0 of 5 folds significant, was 5 of 5), but AC trained T0 on Minnie65 only. AB5 had T0 on Minnie65 + H01 with the OLD
rem direction fit (shared scale).
rem AE1 T0 with the new t0.yaml on Minnie65 + H01 (species in the design); P0 and P1 stay the AB5 runs (nothing in P changed).
rem AE2 generation + evaluation per dataset on val (as AB5) and neuron CV of T0 on the two datasets (4 vs 3 folds of 5 had
rem     tortuosity significant in AB5 / AA, check whether the new fit changes it).
rem AE3 MoGen v4 (AB6) generation: species x binary cell type, guidance 1.25, 300 samples each, then the d8_pw8 Poisson chain.
rem     Evaluation of v4 (mouse against Minnie65 train spines of the same type, human against H01) needs the real controls and
rem     the conditional report; do it after looking at these meshes (the condition effect on the losses was only 0.0008).
rem Log: scripts\neuron_model\overnight\2026-11-20_AE_t0-two-datasets-v4-generation.bat > logs_AE.txt 2>&1

set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set B=C:\Users\Student\LipsEkaterina\data\minnie65\preprocessed_branches
set H01B=C:\Users\Student\LipsEkaterina\data\h01\preprocessed_branches
set SPLITS=data\processed\neuron-model\splits
set SPLITH=%SPLITS%\splits_v1_h01.parquet
set RUNS=runs\training\neuron-model
set ANA=runs\analysis\neuron-model
set CFG=configs\neuron-model\reconstruction\poisson_sweep
set P0H=%RUNS%\2026-11-17_071639_p0
set P1H=%RUNS%\2026-11-17_072620_p1
set V4=%RUNS%\2026-11-17_085545_mogen-conditional-v4-h01
set TPH=%ANA%\tp_generation\minnie_h01_acfit
set ROOTS=%B%;%H01B%

cd /d %REPO%
if not exist %P1H% (echo NO P1 RUN & goto :end)

echo ===== [AE1] T0 (radius-dependent wiggle scale) on Minnie65 + H01 ===== %date% %time%
python scripts\neuron_model\fit_t0.py --config configs\neuron-model\trunk\t0.yaml "data.branches_root=%ROOTS%" data.splits=%SPLITH% "model.mean_models=[parabola_mix]"
if errorlevel 1 (echo T0 FAILED & goto :end)
for /f "delims=" %%d in ('dir /b /ad /o-n %RUNS%\*_t0') do (set T0H=%RUNS%\%%d& goto :t0found)
:t0found
echo new T0 run: %T0H%
python scripts\neuron_model\diagnose_t0_direction.py --t0-run %T0H% "data.branches_root=%ROOTS%" data.splits=%SPLITH%

echo ===== [AE2] generation, evaluation, neuron CV ===== %date% %time%
for %%k in (minnie65 h01) do (
  python scripts\neuron_model\generate_tp.py --t0-run %T0H% --p0-run %P1H% --mode generated --n 1000 --seed 1 --split val --dataset %%k --out-root %TPH%\%%k\generated
  python scripts\neuron_model\generate_tp.py --t0-run %T0H% --p0-run %P1H% --mode real_t --n 1000 --seed 1 --split val --dataset %%k --out-root %TPH%\%%k\real_t
)
for %%k in (minnie65 h01) do (
  for %%m in (generated real_t) do call :evalnewest %TPH%\%%k\%%m
)
python scripts\neuron_model\cv_tp_neurons.py --branches-root "%ROOTS%" --splits %SPLITH% --k 5 --repeats 1 --models t0

echo ===== [AE3] MoGen v4 generation (GPU) ===== %date% %time%
if not exist %V4% (echo NO v4 RUN & goto :end)
for %%s in (mouse human) do (
  for %%c in (excitatory inhibitory) do (
    python scripts\neuron_model\generate_mogen.py --run %V4% --checkpoint latest --condition species=%%s,cell_type_binary=%%c --guidance 1.25 --n-samples 300 --seed 1 --tag v4_%%s_%%c_g1.25_seed_1
    python scripts\neuron_model\reconstruct_mogen_poisson.py --run %V4% --set v4_%%s_%%c_g1.25_seed_1 --out %V4%\meshes_poisson_d8_pw8 --poisson-config %CFG%\d8_pw8.yaml --workers 8
  )
)

:end
echo ===== chain AE done ===== %date% %time%
goto :eof

:evalnewest
for /f "delims=" %%d in ('dir /b /ad /o-n %1') do (
  python scripts\neuron_model\evaluate_assembled_branches.py --tp-dir %1\%%d
  goto :eof
)
goto :eof
