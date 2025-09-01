param(
  [Parameter(Mandatory=$true)][ValidateSet('simulate','train-deep','train-bayes','train-gam','predict-deep','predict-bayes','predict-gam','test')]
  [string]$cmd,
  [string]$Python = "python",
  [string]$DataCfg = "configs/data.yaml",
  [string]$DeepCfg = "configs/deep.yaml",
  [string]$BayesCfg = "configs/bayes.yaml",
  [string]$GamCfg = "configs/gam.yaml",
  [string]$SaveDir = "artifacts",
  [string]$OutDir = "data"
)

$Env:PYTHONPATH = Join-Path (Get-Location) "src"

switch ($cmd) {
  'simulate'       { & $Python "scripts/simulate_data.py" "--out" $OutDir "--config" $DataCfg }
  'train-deep'     { & $Python "scripts/train.py" "--backend" "deep" "--data_cfg" $DataCfg "--train_cfg" $DeepCfg "--save_dir" $SaveDir }
  'train-bayes'    { & $Python "scripts/train.py" "--backend" "bayes" "--data_cfg" $DataCfg "--train_cfg" $BayesCfg "--save_dir" $SaveDir }
  'train-gam'      { & $Python "scripts/train.py" "--backend" "gam" "--data_cfg" $DataCfg "--train_cfg" $GamCfg "--save_dir" $SaveDir }
  'predict-deep'   { & $Python "scripts/predict_ps.py" "--backend" "deep" "--model_path" (Join-Path $SaveDir "deep_cox_binned.pt") "--data_cfg" $DataCfg "--out" (Join-Path $OutDir "ps_deep.parquet") "--export_embeddings" }
  'predict-bayes'  { & $Python "scripts/predict_ps.py" "--backend" "bayes" "--model_path" (Join-Path $SaveDir "bayes_cox_tvf.pkl") "--data_cfg" $DataCfg "--train_cfg" $BayesCfg "--out" (Join-Path $OutDir "ps_bayes.parquet") }
  'predict-gam'    { & $Python "scripts/predict_ps.py" "--backend" "gam" "--model_path" (Join-Path $SaveDir "gam_cox_tvf.pkl") "--data_cfg" $DataCfg "--train_cfg" $GamCfg "--out" (Join-Path $OutDir "ps_gam.parquet") }
  'test'           { & $Python "-m" "pytest" "-q" }
  default          { Write-Error "Unknown command $cmd" }
}
