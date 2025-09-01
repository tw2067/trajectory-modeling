# -------- Project-wide defaults --------
PYTHON      ?= python
DATA_CFG    ?= configs/data.yaml
DEEP_CFG    ?= configs/deep.yaml
BAYES_CFG   ?= configs/bayes.yaml
GAM_CFG     ?= configs/gam.yaml
SAVE_DIR    ?= artifacts
OUT_DIR     ?= data
PYTHONPATH  := src

export PYTHONPATH

# -------- Meta --------
.PHONY: help
help:
	@echo "TrajPS — common commands"
	@echo ""
	@echo "  make setup-min          - install minimal deps for CI (numpy, pandas, lifelines, pytest)"
	@echo "  make setup-deep         - install deep extras (torch, tqdm)"
	@echo "  make setup-bayes        - install bayes extras (pymc, arviz, patsy, joblib, lifelines)"
	@echo "  make setup-gam          - install gam extras (pygam, scipy<=1.10.1, joblib, lifelines)"
	@echo ""
	@echo "  make simulate           - write synthetic dynamic/static parquet to ./data/"
	@echo "  make train-deep         - train deep backend (checkpoint in ./artifacts/)"
	@echo "  make train-bayes        - fit Cox TVF with Bayesian traj covariates"
	@echo "  make train-gam          - fit Cox TVF with GAM traj covariates"
	@echo ""
	@echo "  make predict-deep       - write PS (and embeddings) for deep backend"
	@echo "  make predict-bayes      - write PS for Bayesian backend"
	@echo "  make predict-gam        - write PS for GAM backend"
	@echo ""
	@echo "  make test               - run pytest (skips missing extras)"
	@echo "  make clean              - remove artifacts/"
	@echo "  make veryclean          - remove artifacts/, data/, __pycache__/"

# -------- Setup (no editable install required) --------
.PHONY: setup-min setup-deep setup-bayes setup-gam
setup-min:
	$(PYTHON) -m pip install -U pip wheel setuptools
	$(PYTHON) -m pip install numpy pandas lifelines pytest pyyaml

setup-deep:
	$(PYTHON) -m pip install torch tqdm

setup-bayes:
	$(PYTHON) -m pip install pymc arviz patsy joblib lifelines "pytensor>=2.18"

setup-gam:
	$(PYTHON) -m pip install "numpy<=1.26.4" "scipy<=1.10.1" pygam joblib lifelines

# -------- Data --------
.PHONY: simulate
simulate:
	$(PYTHON) scripts/simulate_data.py --out $(OUT_DIR) --config $(DATA_CFG)

# -------- Train --------
.PHONY: train-deep train-bayes train-gam
train-deep:
	$(PYTHON) scripts/train.py --backend deep --data_cfg $(DATA_CFG) --train_cfg $(DEEP_CFG) --save_dir $(SAVE_DIR)

train-bayes:
	$(PYTHON) scripts/train.py --backend bayes --data_cfg $(DATA_CFG) --train_cfg $(BAYES_CFG) --save_dir $(SAVE_DIR)

train-gam:
	$(PYTHON) scripts/train.py --backend gam --data_cfg $(DATA_CFG) --train_cfg $(GAM_CFG) --save_dir $(SAVE_DIR)

# -------- Predict --------
.PHONY: predict-deep predict-bayes predict-gam
predict-deep:
	$(PYTHON) scripts/predict_ps.py --backend deep --model_path $(SAVE_DIR)/deep_cox_binned.pt \
		--data_cfg $(DATA_CFG) --out $(OUT_DIR)/ps_deep.parquet --export_embeddings

predict-bayes:
	$(PYTHON) scripts/predict_ps.py --backend bayes --model_path $(SAVE_DIR)/bayes_cox_tvf.pkl \
		--data_cfg $(DATA_CFG) --train_cfg $(BAYES_CFG) --out $(OUT_DIR)/ps_bayes.parquet

predict-gam:
	$(PYTHON) scripts/predict_ps.py --backend gam --model_path $(SAVE_DIR)/gam_cox_tvf.pkl \
		--data_cfg $(DATA_CFG) --train_cfg $(GAM_CFG) --out $(OUT_DIR)/ps_gam.parquet

# -------- Tests --------
.PHONY: test
test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q

# -------- Cleanup --------
.PHONY: clean veryclean
clean:
	rm -rf $(SAVE_DIR)

veryclean: clean
	rm -rf $(OUT_DIR) __pycache__ */__pycache__
