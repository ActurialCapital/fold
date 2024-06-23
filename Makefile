.PHONY: docs clean
.EXPORT_ALL_VARIABLES:

clean:
	@echo "Clearning folders"
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name '.DS_Store' -delete
	rm -rf .pytest_cache

activate:
	@echo "Activating virtual environment"
	poetry shell

setup:
	@echo "Installing dependencies with Poetry"
    poetry lock && poetry install && poetry shell

test:
	pytest tests -v