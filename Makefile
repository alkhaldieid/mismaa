# Mismaa developer tasks. Uses the in-repo virtualenv so the pinned
# requirements.txt toolchain is always what runs.

PY  := .venv/bin/python
RUFF := .venv/bin/ruff

.PHONY: test lint fmt

test:            ## Run the DSP test suite
	$(PY) -m pytest -q

lint:            ## Static-check + import-order (no changes written)
	$(RUFF) check .

fmt:             ## Auto-format in place
	$(RUFF) format .
