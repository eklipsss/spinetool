@echo off
rem Chain G2 (GPU sampling + CPU meshing, ~2 h ALONE on the GPU). The reduced version of chain G: only the two sets the conditional-generation
rem report really needs at the working guidance w = 1.25 with 500 samples each. Why reduced: running training and generation on one GPU at the same
rem time collapsed both (VRAM full, WDDM paging): chain G reached 8 point clouds per 960 s instead of 8 per 26 s, the VAE 0.8 steps/s instead of 3.3.
rem Everything else is taken from sets that already exist in the v3 run folder (200 samples each, chains C and E):
rem   unconditional: seed_1 (500, chain G1);  w = 1 / 1.5 / 2 / 3 (dose-response): cond_cell_type_binary-<type>_g<w>_seed_1;
rem   detailed types at w = 1.5: cond_cell_type-<type>_g1.5_seed_1 (4P, 6P-CT, BPC, MC).
rem Run it ALONE on the GPU (no training at the same time), or at most next to a training run if its speed stays above ~1.4 steps/s.
rem Output tags carry _n500 so that the partial folders of the stopped chain G (cond_..._g1.25_seed_1) are not mixed in.
rem Log: scripts\neuron_model\overnight\2026-11-04_G2_condgen-125-only.bat > logs_G2.txt 2>&1

set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set V3=runs\training\neuron-model\2026-11-03_015237_mogen-conditional-v3

cd /d %REPO%

for %%c in (excitatory inhibitory) do (
  echo ===== %%c, guidance 1.25, 500 samples ===== %date% %time%
  python scripts\neuron_model\generate_mogen.py --run %V3% --checkpoint latest --condition cell_type_binary=%%c --guidance 1.25 --n-samples 500 --seed 1 --tag cond_cell_type_binary-%%c_g1.25_seed_1_n500
)

echo ===== G2 done ===== %date% %time%
