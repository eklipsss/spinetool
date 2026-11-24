@echo off
rem 2026-11-26, chain AM. Does the SAMPLING make the generated spines thick? GPU ~2 h, CPU ~10 min. Can run in parallel with AK / AL
rem (they are CPU only).
rem
rem Chains AJ / AJ2: the point cloud of model A is thicker than the real one (hull volume x1.16-1.22, distance from the axis
rem x1.07-1.12) while the length is right; along the fine-tuning x5.8 (mouse_mixed) -> x2.6 (1000) -> x1.28 (3000) -> x1.15 (8000)
rem -> x1.21 (16000): a plateau, longer training does not remove it. Before changing the training, two cheap checks of the
rem sampler on the SAME model A (latest, seed 1, 300 clouds, no Poisson):
rem   number of ODE steps 50 / 200 / 400 (100 = the set seed_1_n500 of chain AF): an integration error would shrink with the steps;
rem   starting noise sd 0.9 / 0.8 at 100 steps: if the excess thickness comes from the spread of the start, it shrinks quickly.
rem Then cloud_size_diagnostics.py against the real clouds (with the radius profile along the spine axis).
rem -> runs\analysis\neuron-model\2026-11-26_sampling-size
rem Log: scripts\neuron_model\overnight\2026-11-26_AM_sampling-size.bat > logs_AM.txt 2>&1

set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set RUNS=runs\training\neuron-model
set ANA=runs\analysis\neuron-model
set MA=%RUNS%\2026-10-20_213103_mogen-ft-dev
set MAN=%MA%\meshes_poisson_d8_pw8_n500
set CMP2=%ANA%\2026-11-21_v3-vs-A-seeds

cd /d %REPO%
if not exist %MAN%\seed_1 (echo NO A seed_1 set & goto :end)

echo ===== [AM1] ODE steps ===== %date% %time%
for %%s in (50 200 400) do (
  if exist %MA%\meshes\steps%%s_seed_1\sample_00299 (echo steps %%s exists, skipped) else (
    python scripts\neuron_model\generate_mogen.py --run %MA% --checkpoint latest --n-samples 300 --seed 1 --steps %%s --tag steps%%s_seed_1 --no-mesh
  )
)
echo ===== [AM2] starting noise sd ===== %date% %time%
for %%n in (0.9 0.8) do (
  if exist %MA%\meshes\noise%%n_seed_1\sample_00299 (echo noise %%n exists, skipped) else (
    python scripts\neuron_model\generate_mogen.py --run %MA% --checkpoint latest --n-samples 300 --seed 1 --noise-scale %%n --tag noise%%n_seed_1 --no-mesh
  )
)

echo ===== [AM3] cloud size ===== %date% %time%
python scripts\neuron_model\cloud_size_diagnostics.py --manifest cache\minnie65_v1_60k\manifest.parquet --real real=%CMP2%\real\poisson\seed_1 ^
  --gen A-steps100=%MAN%\seed_1 --gen A-steps50=%MA%\meshes\steps50_seed_1 --gen A-steps200=%MA%\meshes\steps200_seed_1 --gen A-steps400=%MA%\meshes\steps400_seed_1 ^
  --gen A-noise0.9=%MA%\meshes\noise0.9_seed_1 --gen A-noise0.8=%MA%\meshes\noise0.8_seed_1 --out %ANA%\2026-11-26_sampling-size

:end
echo ===== chain AM done ===== %date% %time%
