@echo off
rem 2026-11-26, chain AL. LabID without the z factor 0.7, split by source image, effect of amyloid, confocal model with the
rem SpineTool last step. CPU only, ~2-3 h (most of it AL6). Independent of the chains AJ / AK (can run after them or on another day).
rem
rem AL1 convert LabID (datasets\labid, branch folders now under <group>\output\) WITHOUT any z factor into a new tree
rem           C:\Users\Student\LipsEkaterina\data\labid\labid_raw_z1_2611 (current code; the old labid_raw_z1 of 24.10 stays)
rem AL2 S preprocessing of that tree (LabID config), AL3 branch preprocessing (LabID config) -> preprocessed_branches_z1_2611
rem AL4 acquisition table of the four groups + split by SOURCE IMAGE (Wt_old left out) -> splits_labid_v1.parquet
rem AL5 group differences, unit = source image: Ab vs Wt (NA 1.40, deconvolved = main), sensitivity, Ab+9009 vs Wt+9009
rem           within one objective, treatment contrasts descriptively -> runs\analysis\neuron-model\2026-11-26_labid-group-effects
rem           (prototype on the mac on the z 1 tables of 24.10: Ab vs Wt volume -28 %, length/volume +31 %, nominal p 0.01-0.04,
rem           q 0.06-0.10 after BH; with the non-deconvolved image Ab 18 4 of 11 metrics significant)
rem AL6 confocal imaging model with the SpineTool last step (8-bit image, Gaussian 1 voxel, local threshold, boundary voxels,
rem           Poisson in voxel units), voxel 22.17 x 22.17 x 100 nm (Leica metadata), PSF 55/180 nm (NA 1.40 oil), thresholds
rem           0.04 / 0.08 / 0.15, and a narrower PSF 40/130 nm at 0.08 as a stand-in for deconvolution (Wt is deconvolved);
rem           500 Minnie65 train spines, LabID Wt (new tree) as the target; decomposition with the source image as the LabID unit
rem           -> runs\analysis\neuron-model\2026-11-26_confocal-spinetool
rem Log: scripts\neuron_model\overnight\2026-11-26_AL_labid-z1-split-effects-spinetool.bat > logs_AL.txt 2>&1

set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set ANA=runs\analysis\neuron-model
set SPLITS=data\processed\neuron-model\splits
set SPLIT=%SPLITS%\splits_v1.parquet
set META=cache\manifest_minnie65_meta.parquet
set LR=C:\Users\Student\LipsEkaterina\data\labid\labid_raw_z1_2611
set LBR=C:\Users\Student\LipsEkaterina\data\labid\preprocessed_branches_z1_2611
set LMAN=%LR%\labid\preprocessed\manifest.parquet
set ACQ=data\interim\neuron-model\labid_acquisition.csv
set CONF=%ANA%\2026-11-26_confocal-spinetool
set VOX=--confocal-param voxel_nm=[22.17,22.17,100] --confocal-param surface=spinetool

cd /d %REPO%
if not exist datasets\labid\Wt\output (echo NO datasets\labid\Wt\output - copy the LabID folder with the new layout & goto :end)

echo ===== [AL1] convert LabID, no z factor ===== %date% %time%
python scripts\neuron_model\convert_labid_layout.py --labid-root datasets\labid --out %LR%
if errorlevel 1 (echo CONVERSION FAILED & goto :end)

echo ===== [AL2] S preprocessing ===== %date% %time%
python scripts\run_spine_preprocessing.py --config data\processed\config\spine_preprocessing_labid.yaml --raw-data-root %LR% --dataset labid --metadata-root %REPO%\datasets
if errorlevel 1 (echo S PREPROCESSING FAILED & goto :end)

echo ===== [AL3] branch preprocessing ===== %date% %time%
python scripts\neuron_model\run_branch_preprocessing.py --dataset labid --config data\processed\config\branch_preprocessing_labid.yaml ^
  --raw-data-root %LR% --metadata-root %REPO%\datasets --s-preprocessed-root %LR%\labid\preprocessed --output-root %LBR%
if errorlevel 1 (echo BRANCH PREPROCESSING FAILED)

echo ===== [AL4] acquisition table and split by source image ===== %date% %time%
python scripts\neuron_model\labid_acquisition_metadata.py --groups Ab Wt Ab+9009 Wt+9009 --out %ACQ% --compare
python scripts\neuron_model\filter_manifest.py --manifest %LMAN% --exclude "neuron_id~^Wt_old" --out %LR%\labid\preprocessed\manifest_no_wt_old.parquet
python scripts\neuron_model\build_splits.py --config configs\neuron-model\data\splits_labid_v1.yaml "manifests=[%LR:\=/%/labid/preprocessed/manifest_no_wt_old.parquet]"

echo ===== [AL5] group differences, unit = source image ===== %date% %time%
python scripts\neuron_model\labid_group_effects.py --morphometrics %LR%\labid\preprocessed\morphometrics.parquet --acquisition %ACQ% --out %ANA%\2026-11-26_labid-group-effects

echo ===== [AL6] confocal model with the SpineTool last step ===== %date% %time%
python scripts\neuron_model\make_real_controls.py --manifest %META% --path-mode absolute --splits %SPLIT% --split train --controls asis confocal ^
  --confocal-param sigma_xy_nm=55 --confocal-param sigma_z_nm=180 --confocal-param threshold=0.08 %VOX% --n-samples 500 --workers 8 --out %CONF%\minnie_n_t0.08
for %%t in (0.04 0.15) do (
  python scripts\neuron_model\make_real_controls.py --manifest %META% --path-mode absolute --splits %SPLIT% --split train --controls confocal ^
    --confocal-param sigma_xy_nm=55 --confocal-param sigma_z_nm=180 --confocal-param threshold=%%t %VOX% --n-samples 500 --workers 8 --out %CONF%\minnie_n_t%%t
)
python scripts\neuron_model\make_real_controls.py --manifest %META% --path-mode absolute --splits %SPLIT% --split train --controls confocal ^
  --confocal-param sigma_xy_nm=40 --confocal-param sigma_z_nm=130 --confocal-param threshold=0.08 %VOX% --n-samples 500 --workers 8 --out %CONF%\minnie_d_t0.08
python scripts\neuron_model\make_real_controls.py --manifest %LMAN% --where "neuron_id~^Wt_[0-9]" --controls asis --n-samples 250 --workers 8 --out %CONF%\labid_wt
python scripts\neuron_model\confocal_domain_shift.py --reference minnie=%CONF%\minnie_n_t0.08\asis\seed_1 --target labid_wt=%CONF%\labid_wt\asis\seed_1 ^
  --imaged st_n_t0.04=%CONF%\minnie_n_t0.04\confocal\seed_1 --imaged st_n_t0.08=%CONF%\minnie_n_t0.08\confocal\seed_1 ^
  --imaged st_n_t0.15=%CONF%\minnie_n_t0.15\confocal\seed_1 --imaged st_d_t0.08=%CONF%\minnie_d_t0.08\confocal\seed_1 --out %CONF%\shift

:end
echo ===== chain AL done ===== %date% %time%
