.PHONY: all build clean test bench grpc

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

grpc:
	@echo "Compiling Protobuf/gRPC schemas..."
	$(PYTHON) -m grpc_tools.protoc -I./optionpricer/api --python_out=./optionpricer/api --grpc_python_out=./optionpricer/api ./optionpricer/api/pricer.proto
