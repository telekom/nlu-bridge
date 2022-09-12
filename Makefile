.DEFAULT_GOAL := help
.PHONY: coverage check help format
src := nlubridge
test-src := tests
other-src := setup.py

check: ## Check code style
	pydocstyle --count $(src) $(test-src) $(other-src)
	black $(src) $(test-src) $(other-src) --check --diff
	flake8 $(src) $(test-src) $(other-src)
	isort $(src) $(test-src) $(other-src) --check --diff
	mdformat --check *.md
	mypy --ignore-missing-imports --install-types --non-interactive $(src) $(test-src) $(other-src)
	pylint --fail-under 6 $(src)

format: ## Auto-format code
	black $(src) $(test-src)
	isort $(src) $(test-src) $(other-src)
	mdformat *.md

test: ## test
	python -m pytest --cov=nlubridge tests

test_publish: ## puplish to testpypi
	python setup.py sdist bdist_wheel
	twine upload -r testpypi dist/*
