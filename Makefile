.PHONY: clean
clean:
	@find . -iname '*.py[co]' -delete
	@find . -iname '__pycache__' -delete
	@rm -rf  '.pytest_cache'
	@rm -rf dist/
	@rm -rf build/
	@rm -rf *.egg-info
	@rm -rf .tox
	@rm -rf venv/lib/python*/site-packages/*.egg
	@rm -rf .ruff_cache/

.PHONY: test-actions
test-actions:
	act -W .github/workflows/ci.yml -W .github/workflows/test.yml

.PHONY: test-actions-mac
test-actions-mac:
	act -W .github/workflows/ci.yml -W .github/workflows/test.yml --container-architecture linux/amd64 -P ubuntu-latest=nektos/act-environments-ubuntu:18.04

.PHONY: install
install:
	poetry install --with dev

.PHONY: test
test:
	poetry run pytest tests/ --cov=src --cov-report=term-missing

.PHONY: lint
lint:
	poetry install --with dev --no-root
	poetry run ruff check src --fix --unsafe-fixes --preview
	poetry run autoflake -r --in-place --remove-unused-variables src/ examples/ tests/
	poetry run black src/ examples/ tests/

.PHONY: cypherpunk-love
cypherpunk-love:
	poetry run python examples/agents/cypherpunk_love.py

.PHONY: world-traveler
world-traveler:
	poetry run python examples/agents/world_traveler.py

.PHONY: summer-chaser
summer-chaser:
	poetry run python examples/agents/summer_chaser.py

.PHONY: dissociative-identity
dissociative-identity:
	poetry run python examples/agents/dissociative_identity.py

.PHONY: cypherpunk-jokes
cypherpunk-jokes:
	poetry run python examples/agents/cypherpunk_jokes.py

.PHONY: friend-with-benefits
friend-with-benefits:
	poetry run python examples/agents/friend_with_benefits.py
