@echo off
rem 2026-11-17, chain AD (CPU, ~1-2 h without the memorization check). Repeats ONLY the evaluation step of AB1.
rem AB1 already produced: splits_v1_val_minus_dev.parquet, the v3 meshes meshes_poisson_d8_pw8 and the real controls
rem (runs\analysis\neuron-model\2026-11-18_v3-vs-A\real\poisson). evaluate_generators.py then stopped: the train cloud cache
rem data\interim\neuron-model\eval\train_clouds_2048.npy was built for another list of train spines (the old cache is NOT touched).
rem
rem Two ways, switch by the MEMO line below:
rem   MEMO=false (default): the Chamfer memorization check against the whole train split is switched off, no cache is needed.
rem                         For the v3-vs-A comparison it is secondary (it was also secondary in the earlier evaluations).
rem   MEMO=true: builds a NEW cache in another file (train_clouds_2048_splits_v1.npy) - one pass over 680 534 clouds from the NAS,
rem              many hours; the old file stays where it is, nothing is overwritten.
rem Log: scripts\neuron_model\overnight\2026-11-19_AD_v3-vs-A-evaluation-only.bat > logs_AD.txt 2>&1

set MEMO=false
set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set SPLITS=data\processed\neuron-model\splits
set RUNS=runs\training\neuron-model
set ANA=runs\analysis\neuron-model
set V3=%RUNS%\2026-11-03_015237_mogen-conditional-v3
set MA=%RUNS%\2026-10-20_213103_mogen-ft-dev
set CMP=%ANA%\2026-11-18_v3-vs-A
set FULL=C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/manifest_minnie65_full.parquet
set MORPH=C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/morphometrics_minnie65_full.parquet

cd /d %REPO%

if not exist %SPLITS%\splits_v1_val_minus_dev.parquet (echo NO SPLIT - run AB1 first & goto :end)
if not exist %V3%\meshes_poisson_d8_pw8 (echo NO v3 MESHES & goto :end)
if not exist %CMP%\real\poisson (echo NO REAL CONTROLS & goto :end)

echo ===== [AD1] evaluation v3 vs A ===== %date% %time%
if "%MEMO%"=="true" (
  python scripts\neuron_model\evaluate_generators.py ^
    --model mogen_v3=%V3%\meshes_poisson_d8_pw8 --model mogen_A=%MA%\meshes_poisson_d8_pw8 --model real_poisson=%CMP%\real\poisson ^
    "data.manifests=['%FULL%']" data.path_mode=absolute data.splits=%SPLITS%\splits_v1_val_minus_dev.parquet ^
    data.bbox=%SPLITS%\bbox_splits_v1.json data.drop_unassigned=true "protocol.generation_seeds=[1]" real.morphometrics_table=%MORPH% ^
    real.train_cloud_cache=data/interim/neuron-model/eval/train_clouds_2048_splits_v1.npy name=generator-evaluation-v3-vs-A
) else (
  python scripts\neuron_model\evaluate_generators.py ^
    --model mogen_v3=%V3%\meshes_poisson_d8_pw8 --model mogen_A=%MA%\meshes_poisson_d8_pw8 --model real_poisson=%CMP%\real\poisson ^
    "data.manifests=['%FULL%']" data.path_mode=absolute data.splits=%SPLITS%\splits_v1_val_minus_dev.parquet ^
    data.bbox=%SPLITS%\bbox_splits_v1.json data.drop_unassigned=true "protocol.generation_seeds=[1]" real.morphometrics_table=%MORPH% ^
    real.chamfer_memorization=false name=generator-evaluation-v3-vs-A
)
if errorlevel 1 (echo EVALUATION FAILED & goto :end)
for /f "delims=" %%d in ('dir /b /ad /o-n %ANA%\*_generator-evaluation-v3-vs-A') do (set EVAL=%ANA%\%%d& goto :evalfound)
:evalfound
echo evaluation run: %EVAL%
python scripts\neuron_model\classifier_ablation.py --eval-run %EVAL% --reference real_poisson --models mogen_v3 mogen_A
python scripts\neuron_model\morphometric_tails.py --eval-run %EVAL% --metrics Volume Area Length ConvexHullVolume
type %EVAL%\postprocessed\summary.md

:end
echo ===== chain AD done ===== %date% %time%
