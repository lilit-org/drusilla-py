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

.PHONY: install
install:
	poetry install --no-root

.PHONY: test
test:
	poetry run pytest

.PHONY: format
format:
	poetry install --with dev --no-root
	poetry run ruff check src --fix --unsafe-fixes
	poetry run autoflake -r --in-place --remove-unused-variables src/ agents_examples/
	poetry run black src/ agents_examples/

.PHONY: cypherpunk-love
cypherpunk-love:
	poetry run python agents_examples/cypherpunk_love.py

.PHONY: world-traveler
world-traveler:
	poetry run python agents_examples/world_traveler.py

.PHONY: summer-chaser
summer-chaser:
	poetry run python agents_examples/summer_chaser.py

.PHONY: dissociative-identity
dissociative-identity:
	poetry run python agents_examples/dissociative_identity.py

.PHONY: cypherpunk-jokes
cypherpunk-jokes:
	poetry run python agents_examples/cypherpunk_jokes.py
