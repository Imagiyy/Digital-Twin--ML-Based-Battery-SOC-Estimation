.PHONY: all setup audit prep train evaluate experiments serve test test-c clean

PYTHON ?= python3

all: setup prep train evaluate experiments test test-c
	@echo "All pipelines, evaluations, ablations, and tests passed successfully!"

setup:
	@echo "Verifying environment dependencies..."
	$(PYTHON) -c "import numpy, scipy, pandas, sklearn, fastapi, uvicorn, matplotlib, yaml, pydantic, pyarrow; print('Environment ready.')"

audit:
	$(PYTHON) -m src.audit

prep:
	$(PYTHON) -m src.data_prep

train:
	$(PYTHON) -m src.train
	$(PYTHON) -m src.mlp_infer

evaluate:
	$(PYTHON) -m src.evaluate

experiments:
	$(PYTHON) -m src.experiments.baselines
	$(PYTHON) -m src.experiments.ablations
	$(PYTHON) -m src.experiments.robustness
	$(PYTHON) -m src.experiments.drift
	$(PYTHON) -m src.experiments.explain

serve:
	$(PYTHON) -m src.server

test:
	$(PYTHON) -m pytest tests/ -v

test-c:
	gcc -O3 -I firmware/include firmware/test/test_c_parity.c -lm -o firmware/test/test_c_parity
	./firmware/test/test_c_parity

clean:
	rm -rf data/processed/*.parquet models/*.joblib models/*.json models/*.h report/figures/*.png report/tables/*.csv
	@echo "Cleaned generated artifacts."
