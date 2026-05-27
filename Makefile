PYTHON     ?= $(shell command -v python3.12 2>/dev/null || command -v python3)
VENV       := .venv
PIP        := $(VENV)/bin/pip
JUPYTER    := $(VENV)/bin/jupyter

.PHONY: setup notebook test lint readme clean help

help: ## Lista los targets disponibles
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

setup: ## Crea .venv, instala dependencias y copia .env.example si no existe .env
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip --quiet
	$(PIP) install -r requirements.txt
	@[ -f .env ] || (cp .env.example .env && echo "AVISO: .env creado — edita tus credenciales antes de correr el pipeline")
	@echo "Listo. Activa el entorno con: source $(VENV)/bin/activate"

notebook: ## Lanza Jupyter Lab en eda/
	$(JUPYTER) lab eda/

test: ## Ejecuta la suite de tests
	$(VENV)/bin/python -m pytest tests/ -v

lint: ## Revisa estilo con ruff (si está instalado en el venv)
	$(VENV)/bin/ruff check src/ --select E,W,F || true

readme: ## Genera README.html (estilo propio) desde README.md con pandoc
	pandoc README.md -s --toc --toc-depth=2 --embed-resources \
		-c docs/readme.css \
		--metadata title="XAI-project · Interpretable Prediction of Mega-Fires in Chile" \
		-o README.html
	@echo "README.html generado. No editar a mano: regenerar con 'make readme'."

clean: ## Elimina .venv y cachés de Python/pytest
	rm -rf $(VENV) .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
