# Commands
SHELL=cmd
PACKAGE_NAME=myproject
EXEC=poetry run

ifeq ($(OS),Windows_NT)
	SHELL=cmd
else
    SHELL=/bin/bash
endif

init:  ## Initialize the development environment.
	pip install -U pip
	pip install -U poetry poetry-dynamic-versioning
shell: init  ## Launch a shell with the project environment activated.
	poetry shell
format:  ## Format the Python files.
	$(EXEC) black $(PACKAGE_NAME)
lab:  ## Launch a Jupyter Lab instance.
	$(EXEC) python -m jupyterlab
format-check:  ## Check code formatting.
	$(EXEC) black --diff --check $(PACKAGE_NAME)
lint-check:  ## Check for common mistakes and anti-patterns.
	$(EXEC) pylint --rcfile pyproject.toml $(PACKAGE_NAME)
type-check:  ## Check for typing issues.
	$(EXEC) mypy --ignore-missing-imports $(PACKAGE_NAME)
check:  ## Run all checks.
	make format-check
	make lint-check
	make type-check
package:  ## Build the wheel and tarball.
	rm -rf dist
	poetry build
