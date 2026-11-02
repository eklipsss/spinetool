@echo off
rem Overnight 2026-11-03, chain D (GPU, light; can run next to chain C). The colleague's two VAE ideas, code of 2026-10-31.
rem   1. analyze_vae_latent: does the code identify the spine; "one spine many times vs many spines" on 40 val spines
rem      (meshes from z = mu, from z ~ q(z|x), from other resamplings; morphometrics; ceiling of any prior)
rem   2. fit_latent_prior: N(0, I) / Gaussian / mixtures / RealNVP flow fitted to the codes of 15 000 train spines,
rem      compared on val codes (NLL, holes, coverage)
rem   3. 200 spines from the flow prior (grid 128, as the VAE of the 2026-10-21 evaluation)
rem   4. evaluate_generators: VAE flow prior next to VAE with the data Gaussian (same protocol as 2026-10-22_031920)
rem Log: scripts\neuron_model\overnight\2026-11-03_D_vae-latent-flow.bat > logs_D.txt 2>&1

set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set VAE=runs\training\neuron-model\2026-10-19_003408_vae-dev-reg0-cos
set FULL=C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/manifest_minnie65_full.parquet

cd /d %REPO%

echo ===== [D1/4] VAE latent analysis ===== %date% %time%
python scripts\neuron_model\analyze_vae_latent.py --run %VAE% --split val --n-spines 1500 --decode-spines 40 --n-posterior 8 --mesh-manifest %FULL%

echo ===== [D2/4] learned priors ===== %date% %time%
python scripts\neuron_model\fit_latent_prior.py --run %VAE% --n-train 15000 --n-val 1500 --gmm 4 8 16

echo ===== [D3/4] spines from the flow prior ===== %date% %time%
python scripts\neuron_model\generate_vae.py --run %VAE% --prior learned --latent-prior %VAE%\latent_prior\flow_prior.pt --n-samples 200 --seed 1 --resolution 128

echo ===== [D4/4] evaluation ===== %date% %time%
python scripts\neuron_model\evaluate_generators.py ^
  --model VAE_flow=%VAE%\meshes_flow_prior --model VAE_data=%VAE%\meshes_data_prior ^
  "data.manifests=['%FULL%']" data.path_mode=absolute ^
  data.splits=data/processed/neuron-model/splits/splits_dev_100n.parquet data.bbox=data/processed/neuron-model/splits/bbox_splits_dev_100n.json ^
  data.drop_unassigned=true "protocol.generation_seeds=[1]" ^
  real.morphometrics_table=C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/morphometrics_minnie65_full.parquet

echo ===== D done ===== %date% %time%
