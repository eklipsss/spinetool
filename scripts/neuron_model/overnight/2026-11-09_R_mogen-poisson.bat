@echo off
rem 2026-11-09, chain R (CPU only, ~2-3 h; can run while chain O is on the GPU): does the better Poisson chain make MoGen closer to real?
rem Evidence (chain P, 2026-11-09_013155_poisson-sweep-val, 200 real dev-VAL spines through the chain): screening point_weight 0 -> 4
rem   lowers the AUC "real vs real through the chain" from 0.857 to 0.734 (depth 8) / 0.709 (depth 9); Area W1/sd 0.22 -> 0.06,
rem   attachment-only AUC 0.83 -> 0.67-0.69, validity 0.90 -> 0.93; depth or k_normal alone change nothing (0.853-0.867).
rem R1 second round of the chain on real spines: point_weight 2 / 8 / 16 at depth 8 (the floor of each variant).
rem R2 the SAME MoGen A / B samples (chain I, seed 1, 200 each; unconditional, so not tied to a split) re-meshed with every variant
rem   into new folders (the evaluated meshes stay untouched).
rem R3 evaluation on the dev VAL neurons: real chains + MoGen with every chain; R4 classifier: everything vs real, and every MoGen
rem   variant vs real spines through THE SAME chain (what is left is the generator). Selection on dev VAL; the untouched test once at the end.
rem Log: scripts\neuron_model\overnight\2026-11-09_R_mogen-poisson.bat > logs_R.txt 2>&1

set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set SPLIT=data\processed\neuron-model\splits\splits_dev_100n_untouched_test.parquet
set CFG=configs\neuron-model\reconstruction\poisson_sweep
set SWEEP=runs\analysis\neuron-model\2026-11-09_real-poisson-sweep
set MA=runs\training\neuron-model\2026-10-20_213103_mogen-ft-dev
set MB=runs\training\neuron-model\2026-10-21_001448_mogen-scratch-dev
set FULL=C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/manifest_minnie65_full.parquet
set MORPH=C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/morphometrics_minnie65_full.parquet

cd /d %REPO%

echo ===== [R1] real val spines through point_weight 2 / 8 / 16 ===== %date% %time%
for %%v in (d8_pw2 d8_pw8 d8_pw16) do (
  echo ----- %%v -----
  python scripts\neuron_model\make_real_controls.py --manifest cache\manifest_minnie65_meta.parquet --path-mode absolute --splits %SPLIT% --split val ^
    --controls poisson --poisson-config %CFG%\%%v.yaml --n-samples 200 --seeds 1 --workers 8 --out %SWEEP%\%%v
)

echo ===== [R2] MoGen A / B re-meshed with every variant ===== %date% %time%
for %%v in (d8_pw2 d8_pw4 d8_pw8 d8_pw16 d9_pw4) do (
  echo ----- %%v -----
  python scripts\neuron_model\reconstruct_mogen_poisson.py --run %MA% --seed 1 --meshes-dir %MA%\meshes --out %MA%\meshes_poisson_%%v --poisson-config %CFG%\%%v.yaml --workers 8
  python scripts\neuron_model\reconstruct_mogen_poisson.py --run %MB% --seed 1 --meshes-dir %MB%\meshes --out %MB%\meshes_poisson_%%v --poisson-config %CFG%\%%v.yaml --workers 8
)

echo ===== [R3] evaluation on the dev VAL neurons ===== %date% %time%
python scripts\neuron_model\evaluate_generators.py ^
  --model real_asis=%SWEEP%\asis\asis --model real_p0=%SWEEP%\p0_current\poisson ^
  --model real_d8_pw2=%SWEEP%\d8_pw2\poisson --model real_d8_pw4=%SWEEP%\d8_pw4\poisson --model real_d8_pw8=%SWEEP%\d8_pw8\poisson ^
  --model real_d8_pw16=%SWEEP%\d8_pw16\poisson --model real_d9_pw4=%SWEEP%\d9_pw4\poisson ^
  --model A_p0=%MA%\meshes --model B_p0=%MB%\meshes ^
  --model A_d8_pw2=%MA%\meshes_poisson_d8_pw2 --model B_d8_pw2=%MB%\meshes_poisson_d8_pw2 ^
  --model A_d8_pw4=%MA%\meshes_poisson_d8_pw4 --model B_d8_pw4=%MB%\meshes_poisson_d8_pw4 ^
  --model A_d8_pw8=%MA%\meshes_poisson_d8_pw8 --model B_d8_pw8=%MB%\meshes_poisson_d8_pw8 ^
  --model A_d8_pw16=%MA%\meshes_poisson_d8_pw16 --model B_d8_pw16=%MB%\meshes_poisson_d8_pw16 ^
  --model A_d9_pw4=%MA%\meshes_poisson_d9_pw4 --model B_d9_pw4=%MB%\meshes_poisson_d9_pw4 ^
  "data.manifests=['%FULL%']" data.path_mode=absolute data.splits=%SPLIT% ^
  data.bbox=data/processed/neuron-model/splits/bbox_splits_dev_100n.json ^
  data.drop_unassigned=true real.test_split=val "protocol.generation_seeds=[1]" real.morphometrics_table=%MORPH% name=mogen-poisson-val

for /f "delims=" %%d in ('dir /b /ad /o-n runs\analysis\neuron-model\*_mogen-poisson-val') do (set EVAL=runs\analysis\neuron-model\%%d& goto :found)
:found
echo evaluation folder: %EVAL%

echo ===== [R4] classifier: vs real, and MoGen vs real through the same chain ===== %date% %time%
python scripts\neuron_model\classifier_ablation.py --eval-run %EVAL%
for %%v in (p0 d8_pw2 d8_pw4 d8_pw8 d8_pw16 d9_pw4) do (
  python scripts\neuron_model\classifier_ablation.py --eval-run %EVAL% --reference real_%%v --models A_%%v B_%%v
)
type %EVAL%\postprocessed\summary.md

echo ===== R done ===== %date% %time%
