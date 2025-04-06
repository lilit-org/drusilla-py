.PHONY: clean
clean:
	@find . -type d -name '__pycache__' -exec rm -rf {} +
	@find . -type f -name '*.py[co]' -delete
	@find . -type d -name '*.egg-info' -exec rm -rf {} +
	@find . -type d -name '.pytest_cache' -exec rm -rf {} +
	@find . -type d -name '.ruff_cache' -exec rm -rf {} +
	@rm -rf dist/ build/ .tox/
	@find venv/lib/python*/site-packages -type d -name '*.egg' -exec rm -rf {} +

.PHONY: install
install:
	poetry install --no-root

.PHONY: test
test:
	poetry run pytest

.PHONY: lint
lint:
	poetry install --with dev --no-root
	poetry run ruff check src --fix --unsafe-fixes
	poetry run autoflake -r --in-place --remove-unused-variables src/ examples/
	poetry run black src/ examples/

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

.PHONY: web-surfer
web-surfer:
	poetry run python examples/agents/web_surfer.py
