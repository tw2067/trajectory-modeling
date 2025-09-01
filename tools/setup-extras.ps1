param(
  [ValidateSet('min','deep','bayes','gam','all')]
  [string]$extra = 'min',
  [string]$Python = 'python'
)

switch ($extra) {
  'min'  { & $Python -m pip install -U pip wheel setuptools; & $Python -m pip install numpy pandas lifelines pytest pyyaml }
  'deep' { & $Python -m pip install torch tqdm }
  'bayes'{ & $Python -m pip install pymc arviz patsy joblib lifelines "pytensor>=2.18" }
  'gam'  { & $Python -m pip install "numpy<=1.26.4" "scipy<=1.10.1" pygam joblib lifelines }
  'all'  {
    & $Python -m pip install -U pip wheel setuptools
    & $Python -m pip install numpy pandas lifelines pytest pyyaml
    & $Python -m pip install torch tqdm
    & $Python -m pip install pymc arviz patsy joblib lifelines "pytensor>=2.18"
    & $Python -m pip install "numpy<=1.26.4" "scipy<=1.10.1" pygam joblib lifelines
  }
}
