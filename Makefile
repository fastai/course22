check: format-check lint type-check

clean: ## Clean the repository
	rm -rf dist
	rm -rf *.egg-info

format: ## Format repository code
	poetry run black *.py

format-check: ## Check the code format
	poetry run black --check *.py

install: artifactory-config ## Install dependencies
	poetry run pip install --upgrade pip
	poetry install -v -E all

lint: ## Launch the linting tool
	poetry run pylint *.py

type-check: ## Launch the type checking tool
	poetry run mypy *.py

update: ## Update python dependencies
	poetry update

help: ## Show the available commands
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
