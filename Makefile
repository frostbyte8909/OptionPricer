.PHONY: all build clean test bench grpc check-version-tag

PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
PYTEST ?= .venv/bin/pytest

all: clean build test grpc

build:
	@echo "Building OptionPricer C-Extensions..."
	$(PYTHON) -m build
	$(PIP) install -e .

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/ dist/ optionpricer.egg-info/
	rm -rf optionpricer/models/*.so optionpricer/models/*.c optionpricer/models/*.o
	find . -type d -name "__pycache__" -exec rm -r {} +

test:
	@echo "Running test suite..."
	$(PYTEST) tests/ -v

bench:
	@echo "Running benchmark suite..."
	$(PYTHON) tests/bench_v2.py

# Before publishing: TAG must be v + exact value of version in pyproject.toml
check-version-tag:
	@test -n "$(TAG)" || (echo "Usage: make check-version-tag TAG=v0.2.1" && exit 1)
	@python3 scripts/verify_release_version.py $(TAG)

grpc:
	@echo "Compiling Protobuf/gRPC schemas..."
	$(PYTHON) -m grpc_tools.protoc -I./optionpricer/api --python_out=./optionpricer/api --grpc_python_out=./optionpricer/api ./optionpricer/api/pricer.proto
