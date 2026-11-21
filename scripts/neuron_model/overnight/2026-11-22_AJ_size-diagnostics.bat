@echo off
rem 2026-11-22, chain AJ (predefense 26.11). Why are generated spines too large? GPU ~2.5 h, CPU ~20 min.
rem Run after git pull, after chain AI (uses its sets). Independent of the other chains.
rem
rem Facts (AF / AI): median volume of generated meshes x1.22 (A) / x1.44 (v3) of the real one, real spines through the
rem same Poisson chain x1.03, length reproduced (x1.0). Two explanations predict different things:
rem   the generator makes fatter shapes -> the POINT CLOUD is already thicker (radial size, hull volume);
rem   the cloud has the right size but is noisier -> Poisson wraps the noise: larger surface noise and mesh / hull ratio.
rem And: does the size bias shrink with fine-tuning steps (then: train longer) or is it inherited from mouse_mixed?
rem
rem AJ1 (GPU) size along the fine-tuning of A: the repeat run of the A recipe with step_N checkpoints
rem           (2026-11-04_095002_mogen-ft-dev-curves): steps 1000, 3000, 8000, 16000 + mouse_mixed zero-shot (step 0),
rem           300 unconditional samples each, seed 1, d8_pw8 Poisson
rem AJ2 (CPU) cloud_size_diagnostics.py: clouds BEFORE Poisson and meshes after, against the real clouds of the same test
rem           spines (real controls of chain AI, seed 1, 300): A seeds 1 and 2, v3 seeds 1 and 2, v4 mouse 23P / 6P-CT g1.5
rem           (descriptive: typed sets vs the untyped real test), the AJ1 sweep
rem           -> runs\analysis\neuron-model\2026-11-22_cloud-size
rem (The paired neuron-level bootstrap A vs v3 is computed on the mac from the AI tables, not here.)
rem Log: scripts\neuron_model\overnight\2026-11-22_AJ_size-diagnostics.bat > logs_AJ.txt 2>&1

set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set RUNS=runs\training\neuron-model
set ANA=runs\analysis\neuron-model
set CFG=configs\neuron-model\reconstruction\poisson_sweep
set V3=%RUNS%\2026-11-03_015237_mogen-conditional-v3
set V4=%RUNS%\2026-11-17_085545_mogen-conditional-v4-h01
set MA=%RUNS%\2026-10-20_213103_mogen-ft-dev
set AC=%RUNS%\2026-11-04_095002_mogen-ft-dev-curves
set P3=%V3%\meshes_poisson_d8_pw8
set P4=%V4%\meshes_poisson_d8_pw8
set MAN=%MA%\meshes_poisson_d8_pw8_n500
set PAC=%AC%\meshes_poisson_d8_pw8
set REAL=%ANA%\2026-11-21_v3-vs-A-seeds\real\poisson\seed_1
set OUT=%ANA%\2026-11-22_cloud-size

cd /d %REPO%
if not exist %AC%\checkpoints (echo NO curves run %AC% & goto :end)
if not exist %REAL%\sample_00299 (echo NO real controls of chain AI & goto :end)
if not exist %MAN%\seed_2 (echo NO A seed_2 of chain AI & goto :end)
echo checkpoints of the curves run:
dir /b %AC%\checkpoints

echo ===== [AJ1] size along the fine-tuning of A (GPU) ===== %date% %time%
for %%s in (1000 3000 8000 16000) do (
  if exist %PAC%\sweep_step%%s\sample_00299 (echo sweep_step%%s already reconstructed, skipped) else (
    if not exist %AC%\checkpoints\step_%%s.pt (echo NO checkpoint step_%%s, skipped) else (
      python scripts\neuron_model\generate_mogen.py --run %AC% --checkpoint step_%%s --n-samples 300 --seed 1 --tag sweep_step%%s --no-mesh
      if errorlevel 1 (echo GENERATION step %%s FAILED) else (
        python scripts\neuron_model\reconstruct_mogen_poisson.py --run %AC% --meshes-dir %AC%\meshes --set sweep_step%%s --out %PAC% --poisson-config %CFG%\d8_pw8.yaml --workers 8
      )
    )
  )
)
if exist %PAC%\sweep_step0\sample_00299 (echo sweep_step0 already reconstructed, skipped) else (
  python scripts\neuron_model\generate_mogen.py --weights data\external\mogen\mouse_mixed_750000_weights.npz --config %MA%\config.yaml --n-samples 300 --seed 1 --tag sweep_step0 --no-mesh
  if errorlevel 1 (echo ZERO-SHOT GENERATION FAILED) else (
    for /f "delims=" %%d in ('dir /b /ad /o-n %ANA%\*_mogen-zero-shot') do (set ZS=%ANA%\%%d& goto :zsfound)
  )
)
goto :zsdone
:zsfound
echo zero-shot run: %ZS%
python scripts\neuron_model\reconstruct_mogen_poisson.py --run %ZS% --meshes-dir %ZS%\meshes --set sweep_step0 --out %PAC% --poisson-config %CFG%\d8_pw8.yaml --workers 8
:zsdone

echo ===== [AJ2] point-cloud size before Poisson (CPU) ===== %date% %time%
python scripts\neuron_model\cloud_size_diagnostics.py --manifest cache\minnie65_v1_60k\manifest.parquet --real real=%REAL% ^
  --gen A-s1=%MAN%\seed_1 --gen A-s2=%MAN%\seed_2 --gen v3-s1=%P3%\seed_1 --gen v3-s2=%P3%\seed_2 ^
  --gen v4-23P-g1.5=%P4%\v4_mouse_23P_g1.5_seed_1 --gen v4-6P-CT-g1.5=%P4%\v4_mouse_6P-CT_g1.5_seed_1 ^
  --gen Acurves-step0=%PAC%\sweep_step0 --gen Acurves-step1000=%PAC%\sweep_step1000 --gen Acurves-step3000=%PAC%\sweep_step3000 ^
  --gen Acurves-step8000=%PAC%\sweep_step8000 --gen Acurves-step16000=%PAC%\sweep_step16000 ^
  --out %OUT%
if errorlevel 1 (echo CLOUD SIZE FAILED)

:end
echo ===== chain AJ done ===== %date% %time%
