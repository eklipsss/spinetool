@echo off
setlocal enabledelayedexpansion
rem 2026-11-26, chain AO. Full quality report of the generation per module + Chamfer memorization of the spine generators.
rem Run AFTER AK (AK0 writes evaluation_paired of the AH sets, AK3 the DRS evaluation) and AN (T0 with the neuron effect).
rem AO1 CPU ~15 min, AO2 GPU + CPU ~2-4 h (the train cloud cache: 20 000 point clouds read once from the NAS).
rem
rem AO0 second generation seed of AN2 (added 2026-11-25 after AN): in AN per_neuron came out worse than independent
rem     (radius +21 nm, spines +4 per branch), and the mean level of the 74 neurons drawn with seed 1 was +0.026
rem     (about 2 standard errors). Same T0 (newest *_t0-neuron), same P1, --seed 2, the three modes, Minnie65 and H01,
rem     evaluate_assembled_branches + diagnose_tp_counts -> tp_generation\minnie_h01_neuron\{minnie65,h01}\generated\*_seed2
rem     (CPU ~30 min). AO1 picks these sets up together with seed 1.
rem AO1 make_quality_report.py (protocol: docs\neuron-model\generation-quality-protocol.md): tables and distribution figures,
rem     real against generated, for
rem       S overall   - evaluation of chain AI (A and v3, seeds 1 + 2, floor = real spines through d8_pw8) and, if present,
rem                     the DRS evaluation of chain AK3;
rem       S per type  - the conditional reports of chain AH with the floor from validation neurons (mouse binary type,
rem                     mouse detailed types, human);
rem       T and P     - the newest AH sets (Minnie65 / H01, full generation and real trunk; paired evaluation of AK0) and the
rem                     sets of chain AN (T0 with the neuron effect, three modes); overall and per cell type / compartment.
rem     -> runs\analysis\neuron-model\2026-11-26_quality-report\quality_report.md (+ PNG, CSV)
rem AO2 the shape memorization (Chamfer DCR, near copies by shape) was switched off in AF / AI (no cache of the ~600 000
rem     training clouds). subsample_split.py: training part = random 20 000 spines of the UNION of the training parts of
rem     splits_v1 (v3) and splits_dev_100n (A), test = val minus dev as in AI; generated and held-out real clouds are
rem     compared with the same subsample. evaluate_generators.py, seed 1, A / v3 / real controls.
rem     -> runs\analysis\neuron-model\<date>_generator-evaluation-memorization
rem Log: scripts\neuron_model\overnight\2026-11-26_AO_quality-report-and-memorization.bat > logs_AO.txt 2>&1

set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set SPLITS=data\processed\neuron-model\splits
set RUNS=runs\training\neuron-model
set ANA=runs\analysis\neuron-model
set V3=%RUNS%\2026-11-03_015237_mogen-conditional-v3
set MA=%RUNS%\2026-10-20_213103_mogen-ft-dev
set P3=%V3%\meshes_poisson_d8_pw8
set MAN=%MA%\meshes_poisson_d8_pw8_n500
set CMP2=%ANA%\2026-11-21_v3-vs-A-seeds
set EVAI=%ANA%\2026-11-21_045820_generator-evaluation-v3-vs-A-n500-2seeds
set TPH=%ANA%\tp_generation\minnie_h01_acfit
set TPN=%ANA%\tp_generation\minnie_h01_neuron
set FULL=C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/manifest_minnie65_full.parquet
set MORPH=C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/morphometrics_minnie65_full.parquet
set QR=%ANA%\2026-11-26_quality-report
set P1H=%RUNS%\2026-11-17_072620_p1

cd /d %REPO%
if not exist %EVAI%\real_test_morphometrics.parquet (echo NO evaluation of chain AI & goto :end)

echo ===== [AO0] AN2 with generation seed 2 ===== %date% %time%
set T0N=
for /f "delims=" %%d in ('dir /b /ad /o-n %RUNS%\*_t0-neuron 2^>nul') do (if not defined T0N set T0N=%RUNS%\%%d)
if not defined T0N (echo NO T0 run of chain AN, AO0 skipped & goto :ao1)
if not exist %P1H%\p0_bundle.json (echo NO P1 run %P1H%, AO0 skipped & goto :ao1)
echo T0 run: %T0N%
for %%k in (minnie65 h01) do (
  for %%m in (independent per_neuron posterior) do (
    python scripts\neuron_model\generate_tp.py --t0-run %T0N% --t0-model parabola_mix_neuron --p0-run %P1H% --mode generated --n 1000 --seed 2 --split val --dataset %%k --t0-neuron %%m --out-root %TPN%\%%k\generated
    if errorlevel 1 (echo GENERATION %%k %%m seed 2 FAILED) else (call :evalnewest %TPN%\%%k\generated)
  )
)

:ao1
echo ===== [AO1] quality report ===== %date% %time%
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
set DRS=
for /f "delims=" %%d in ('dir /b /ad /o-n %ANA%\*_generator-evaluation-drs-seed2 2^>nul') do (if not defined DRS set DRS=%ANA%\%%d)
echo T/P sets: !TP!
python scripts\neuron_model\make_quality_report.py --out %QR% --s-eval %EVAI% ^
  --s-conditional %ANA%\2026-11-20_conditional-report-v4-mouse-valfloor %ANA%\2026-11-20_conditional-report-v4-types-valfloor %ANA%\2026-11-20_conditional-report-v4-human-valfloor ^
  --tp !TP!
if defined DRS python scripts\neuron_model\make_quality_report.py --out %QR%\drs --s-eval !DRS!

echo ===== [AO2] Chamfer memorization ===== %date% %time%
set EXTRA=
if exist %SPLITS%\splits_dev_100n.parquet set EXTRA=--part-from %SPLITS%\splits_dev_100n.parquet
python scripts\neuron_model\subsample_split.py --splits %SPLITS%\splits_v1_val_minus_dev.parquet --part train --n 20000 --keep test %EXTRA% ^
  --out %SPLITS%\splits_v1_val_minus_dev_train20k.parquet
if errorlevel 1 (echo SPLIT FAILED & goto :end)
python scripts\neuron_model\evaluate_generators.py ^
  --model mogen_v3=%P3% --model mogen_A=%MAN% --model real_poisson=%CMP2%\real\poisson ^
  "data.manifests=['%FULL%']" data.path_mode=absolute data.splits=%SPLITS%\splits_v1_val_minus_dev_train20k.parquet ^
  data.bbox=%SPLITS%\bbox_splits_v1.json data.drop_unassigned=true "protocol.generation_seeds=[1]" real.morphometrics_table=%MORPH% ^
  real.chamfer_memorization=true real.train_cloud_cache=data/interim/neuron-model/eval/train_clouds_2048_train20k.npy name=generator-evaluation-memorization
for /f "delims=" %%d in ('dir /b /ad /o-n %ANA%\*_generator-evaluation-memorization') do (set MEM=%ANA%\%%d& goto :memfound)
:memfound
type %MEM%\postprocessed\summary.md

:end
echo ===== chain AO done ===== %date% %time%
goto :eof

:evalnewest
for /f "delims=" %%d in ('dir /b /ad /o-n %1') do (
  python scripts\neuron_model\evaluate_assembled_branches.py --tp-dir %1\%%d
  python scripts\neuron_model\diagnose_tp_counts.py --tp-dir %1\%%d
  goto :eof
)
goto :eof

:addnewest
for /f "delims=" %%d in ('dir /b /ad /o-n %1 2^>nul') do (
  if exist %1\%%d\evaluation_paired\evaluation_branches.csv set TP=!TP! %1\%%d@%2
  goto :eof
)
goto :eof
