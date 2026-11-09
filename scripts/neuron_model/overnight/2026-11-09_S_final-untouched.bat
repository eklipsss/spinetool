@echo off
rem 2026-11-09, chain S (CPU only, ~9-10 h like R3): the final comparison of the S models on the UNTOUCHED test neurons.
rem Choice made on the dev VAL neurons only (chain R, 2026-11-09_114917_mogen-poisson-val; real val spines through the chain vs
rem   the same real spines, AUC: point_weight 0 0.857, 2 0.774, 4 0.734, 8 0.712, 16 0.699, depth 9 + pw 4 0.709):
rem   main chain = depth 8, point_weight 8 (on the plateau; 16 is not better beyond noise and slightly less valid),
rem   sensitivity = depth 8, point_weight 16. Declared BEFORE this run; the test is not used to choose between them.
rem The test split was looked at once before (chain I, 2026-11-06_112409, old chain); nothing was tuned on it since.
rem S0 tails of the morphometrics on dev val (MoGen B: Volume W1/sd 3.1-3.2 at a median shift +0.28 sd) - print only.
rem S1 the same 200 real test spines as the controls of chain I (same split, n, seed) through d8_pw8 and d8_pw16.
rem S2 one evaluation: MoGen A / B (the samples of chain I re-meshed in chain R), VAE_flow (model C), real controls.
rem S3 classifier: vs real; generators vs real spines through THEIR OWN chain.
rem Log: scripts\neuron_model\overnight\2026-11-09_S_final-untouched.bat > logs_S.txt 2>&1

set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set SPLIT=data\processed\neuron-model\splits\splits_dev_100n_untouched_test.parquet
set CFG=configs\neuron-model\reconstruction\poisson_sweep
set RVAL=runs\analysis\neuron-model\2026-11-09_114917_mogen-poisson-val
set CTRL=runs\analysis\neuron-model\2026-11-06_004118_real-controls
set FINAL=runs\analysis\neuron-model\2026-11-09_real-controls-poisson-final
set VAE=runs\training\neuron-model\2026-10-19_003408_vae-dev-reg0-cos
set MA=runs\training\neuron-model\2026-10-20_213103_mogen-ft-dev
set MB=runs\training\neuron-model\2026-10-21_001448_mogen-scratch-dev
set FULL=C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/manifest_minnie65_full.parquet
set MORPH=C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/morphometrics_minnie65_full.parquet

cd /d %REPO%

echo ===== [S0] tails on dev val (chain R) ===== %date% %time%
python scripts\neuron_model\morphometric_tails.py --eval-run %RVAL% --models real_d8_pw8 A_d8_pw8 B_d8_pw8 A_d8_pw16 B_d8_pw16 A_p0 B_p0 --metrics Volume Area Length ConvexHullVolume

echo ===== [S1] real test spines through d8_pw8 / d8_pw16 ===== %date% %time%
for %%v in (d8_pw8 d8_pw16) do (
  echo ----- %%v -----
  python scripts\neuron_model\make_real_controls.py --manifest cache\manifest_minnie65_meta.parquet --path-mode absolute --splits %SPLIT% --split test ^
    --controls poisson --poisson-config %CFG%\%%v.yaml --n-samples 200 --seeds 1 --workers 8 --out %FINAL%\%%v
)

echo ===== [S2] evaluation on the untouched test neurons ===== %date% %time%
python scripts\neuron_model\evaluate_generators.py ^
  --model mogen_A=%MA%\meshes_poisson_d8_pw8 --model mogen_B=%MB%\meshes_poisson_d8_pw8 ^
  --model mogen_A_pw16=%MA%\meshes_poisson_d8_pw16 --model mogen_B_pw16=%MB%\meshes_poisson_d8_pw16 ^
  --model VAE_flow=%VAE%\meshes_flow_prior ^
  --model real_asis=%CTRL%\asis --model real_poisson=%FINAL%\d8_pw8\poisson --model real_poisson_pw16=%FINAL%\d8_pw16\poisson ^
  --model real_mc=%CTRL%\mc ^
  "data.manifests=['%FULL%']" data.path_mode=absolute data.splits=%SPLIT% ^
  data.bbox=data/processed/neuron-model/splits/bbox_splits_dev_100n.json ^
  data.drop_unassigned=true "protocol.generation_seeds=[1]" real.morphometrics_table=%MORPH% name=generator-evaluation-final

for /f "delims=" %%d in ('dir /b /ad /o-n runs\analysis\neuron-model\*_generator-evaluation-final') do (set EVAL=runs\analysis\neuron-model\%%d& goto :found)
:found
echo evaluation folder: %EVAL%

echo ===== [S3] classifier ===== %date% %time%
python scripts\neuron_model\classifier_ablation.py --eval-run %EVAL%
python scripts\neuron_model\classifier_ablation.py --eval-run %EVAL% --reference real_poisson --models mogen_A mogen_B
python scripts\neuron_model\classifier_ablation.py --eval-run %EVAL% --reference real_poisson_pw16 --models mogen_A_pw16 mogen_B_pw16
python scripts\neuron_model\classifier_ablation.py --eval-run %EVAL% --reference real_mc --models VAE_flow
python scripts\neuron_model\classifier_ablation.py --eval-run %EVAL% --reference real_asis --models real_poisson real_poisson_pw16 real_mc
python scripts\neuron_model\morphometric_tails.py --eval-run %EVAL% --metrics Volume Area Length ConvexHullVolume
type %EVAL%\postprocessed\summary.md

echo ===== S done ===== %date% %time%
