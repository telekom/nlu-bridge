.DEFAULT_GOAL := help
.PHONY: coverage check help format
src := nlubridge
test-src := tests
other-src := setup.py

check: ## Check code style
	# pydocstyle --count $(src) $(test-src) $(other-src)
	black $(src) $(test-src) --check #--diff
	flake8 $(src) $(test-src)
	# isort $(src) $(test-src) $(other-src) --check --diff
	# mdformat --check *.md
	# mypy --install-types --non-interactive $(src) $(test-src) $(other-src)
	# pylint $(src)

format: ## Auto-format code
	black $(src) $(test-src)
	# isort $(src) $(test-src) $(other-src)
	# mdformat *.md

test: ## test
	python -m pytest --cov=nluvendors tests/

publish_test: ## publish to TestPyPI
	flit publish --repository pypitest

publish: ## publish to TestPyPI
	flit publish