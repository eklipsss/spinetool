@echo off
rem Overnight 2026-11-09, chain P (CPU only, ~2-3 h; runs IN PARALLEL with the GPU chain O): calibration of the Screened Poisson chain of MoGen.
rem Evidence: configs\neuron-model\reconstruction\poisson.yaml was fitted on 25 spines of a pilot (depth 7, point_weight 0, UNFROZEN, did not
rem   reach the validity bar 0.99). On the untouched test, REAL spines through this chain are told apart from the real ones with AUC 0.87
rem   (as-is: 0.54), MoGen against them only 0.58: the visible gap of MoGen is the surface chain, not the generator (chain I).
rem Here 200 REAL dev-VAL spines go through the chain with one parameter changed at a time (depth 8 / 9, point_weight 4 = screening,
rem   k_normal 15 / 60), then they are evaluated as generators: real_poisson variants against the real val spines. Lower classifier
rem   AUC / MMD and higher validity = a better chain. Selection on dev VAL; the untouched test is not used.
rem The surface chain only: normals are estimated from the points, as for MoGen outputs (no true normals).
rem Log: scripts\neuron_model\overnight\2026-11-09_P_poisson-sweep.bat > logs_P.txt 2>&1

set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set SPLIT=data\processed\neuron-model\splits\splits_dev_100n_untouched_test.parquet
set CTRLV=runs\analysis\neuron-model\2026-11-08_real-controls-val
set CFG=configs\neuron-model\reconstruction\poisson_sweep
set OUT=runs\analysis\neuron-model\2026-11-09_real-poisson-sweep
set FULL=C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/manifest_minnie65_full.parquet
set MORPH=C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/morphometrics_minnie65_full.parquet

cd /d %REPO%

echo ===== [P1] as-is control and the Poisson variants ===== %date% %time%
python scripts\neuron_model\make_real_controls.py --manifest cache\manifest_minnie65_meta.parquet --path-mode absolute --splits %SPLIT% --split val ^
  --controls asis --n-samples 200 --seeds 1 --workers 4 --out %OUT%\asis
for %%v in (p0_current d8 d9 d8_pw4 d9_pw4 d8_k15 d8_k60) do (
  echo ----- %%v ----- %date% %time%
  python scripts\neuron_model\make_real_controls.py --manifest cache\manifest_minnie65_meta.parquet --path-mode absolute --splits %SPLIT% --split val ^
    --controls poisson --poisson-config %CFG%\%%v.yaml --n-samples 200 --seeds 1 --workers 4 --out %OUT%\%%v
)

echo ===== [P2] evaluation on the dev VAL neurons ===== %date% %time%
python scripts\neuron_model\evaluate_generators.py ^
  --model real_asis=%OUT%\asis\asis ^
  --model p0_current=%OUT%\p0_current\poisson --model d8=%OUT%\d8\poisson --model d9=%OUT%\d9\poisson ^
  --model d8_pw4=%OUT%\d8_pw4\poisson --model d9_pw4=%OUT%\d9_pw4\poisson ^
  --model d8_k15=%OUT%\d8_k15\poisson --model d8_k60=%OUT%\d8_k60\poisson ^
  --model real_mc=%CTRLV%\mc ^
  "data.manifests=['%FULL%']" data.path_mode=absolute data.splits=%SPLIT% ^
  data.bbox=data/processed/neuron-model/splits/bbox_splits_dev_100n.json ^
  data.drop_unassigned=true real.test_split=val "protocol.generation_seeds=[1]" real.morphometrics_table=%MORPH% name=poisson-sweep-val

for /f "delims=" %%d in ('dir /b /ad /o-n runs\analysis\neuron-model\*_poisson-sweep-val') do (set EVAL=runs\analysis\neuron-model\%%d& goto :found)
:found
echo evaluation folder: %EVAL%
python scripts\neuron_model\classifier_ablation.py --eval-run %EVAL%
type %EVAL%\postprocessed\summary.md

echo ===== P done ===== %date% %time%
