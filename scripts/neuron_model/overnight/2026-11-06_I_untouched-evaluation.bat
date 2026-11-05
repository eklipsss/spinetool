@echo off
rem Overnight 2026-11-06, chain I (replaces the crashed evaluation of window 1 and step H3/H4 of chain H).
rem Both crashed in rbf_mmd2: the untouched test split has 69 520 real spines -> a 36 GB kernel matrix (fixed 2026-11-06:
rem block sums + a fixed seeded subset of 15 000 real spines, the same for every model).
rem ONE evaluation of every set on the untouched test split (same real subset, same protocol), then the classifier ablation:
rem   generators: MoGen A, MoGen B, VAE + flow prior (128^3 and 256^3, the same 200 latents), VAE + data Gaussian
rem   real controls (the same 200 real test spines): as is, Poisson chain (MoGen floor), Marching Cubes 128^3 / 256^3 (VAE floor)
rem Log: scripts\neuron_model\overnight\2026-11-06_I_untouched-evaluation.bat > logs_I.txt 2>&1

set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set VAE=runs\training\neuron-model\2026-10-19_003408_vae-dev-reg0-cos
set CTRL=runs\analysis\neuron-model\2026-11-06_004118_real-controls
set CTRL256=runs\analysis\neuron-model\2026-11-06_real-controls-r256
set FULL=C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/manifest_minnie65_full.parquet
set MORPH=C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/morphometrics_minnie65_full.parquet

cd /d %REPO%

echo ===== [I1/2] evaluation of 9 sets ===== %date% %time%
python scripts\neuron_model\evaluate_generators.py ^
  --model mogen_A=runs\training\neuron-model\2026-10-20_213103_mogen-ft-dev\meshes ^
  --model mogen_B=runs\training\neuron-model\2026-10-21_001448_mogen-scratch-dev\meshes ^
  --model VAE_flow=%VAE%\meshes_flow_prior --model VAE_flow_r256=%VAE%\meshes_flow_prior_r256 --model VAE_data=%VAE%\meshes_data_prior ^
  --model real_asis=%CTRL%\asis --model real_poisson=%CTRL%\poisson --model real_mc=%CTRL%\mc --model real_mc_r256=%CTRL256%\mc ^
  "data.manifests=['%FULL%']" data.path_mode=absolute ^
  data.splits=data/processed/neuron-model/splits/splits_dev_100n_untouched_test.parquet ^
  data.bbox=data/processed/neuron-model/splits/bbox_splits_dev_100n.json ^
  data.drop_unassigned=true "protocol.generation_seeds=[1]" real.morphometrics_table=%MORPH% name=generator-evaluation-untouched

echo ===== [I2/2] classifier ablation ===== %date% %time%
for /f "delims=" %%d in ('dir /b /ad /o-n runs\analysis\neuron-model\*_generator-evaluation-untouched') do (set EVAL=runs\analysis\neuron-model\%%d& goto :found)
:found
echo evaluation folder: %EVAL%
rem every set against the real test spines
python scripts\neuron_model\classifier_ablation.py --eval-run %EVAL%
rem generators against real spines through THEIR OWN surface chain (what is left is the generator)
python scripts\neuron_model\classifier_ablation.py --eval-run %EVAL% --reference real_poisson --models mogen_A mogen_B
python scripts\neuron_model\classifier_ablation.py --eval-run %EVAL% --reference real_mc --models VAE_flow VAE_data
python scripts\neuron_model\classifier_ablation.py --eval-run %EVAL% --reference real_mc_r256 --models VAE_flow_r256
rem the chains themselves against real spines whose attachment is found by the same heuristic (what the chain alone adds)
python scripts\neuron_model\classifier_ablation.py --eval-run %EVAL% --reference real_asis --models real_poisson real_mc real_mc_r256

echo ===== I done ===== %date% %time%
