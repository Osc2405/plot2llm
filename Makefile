.PHONY: help install install-dev test test-cov lint format clean build publish docs docker-build docker-test docker-jupyter

# Variables
PYTHON = python
PIP = pip
PACKAGE_NAME = plot2llm
VERSION = 0.1.20

# Colores para output
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[1;33m
BLUE = \033[0;34m
NC = \033[0m # No Color

help: ## Mostrar esta ayuda
	@echo "$(BLUE)Plot2LLM - Comandos disponibles:$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""

install: ## Instalar plot2llm
	@echo "$(YELLOW)Instalando plot2llm...$(NC)"
	$(PIP) install -e .

install-dev: ## Instalar plot2llm con dependencias de desarrollo
	@echo "$(YELLOW)Instalando plot2llm con dependencias de desarrollo...$(NC)"
	$(PIP) install -e ".[dev,all]"

install-all: ## Instalar plot2llm con todas las dependencias
	@echo "$(YELLOW)Instalando plot2llm con todas las dependencias...$(NC)"
	$(PIP) install -e ".[all]"

test: ## Ejecutar tests
	@echo "$(YELLOW)Ejecutando tests...$(NC)"
	$(PYTHON) -m pytest tests/ -v

test-cov: ## Ejecutar tests con cobertura
	@echo "$(YELLOW)Ejecutando tests con cobertura...$(NC)"
	$(PYTHON) -m pytest tests/ -v --cov=$(PACKAGE_NAME) --cov-report=html --cov-report=term

test-fast: ## Ejecutar tests rápidos (sin cobertura)
	@echo "$(YELLOW)Ejecutando tests rápidos...$(NC)"
	$(PYTHON) -m pytest tests/ -v -x

test-slow: ## Ejecutar tests lentos
	@echo "$(YELLOW)Ejecutando tests lentos...$(NC)"
	$(PYTHON) -m pytest tests/ -v -m "slow"

test-matplotlib: ## Ejecutar solo tests de matplotlib
	@echo "$(YELLOW)Ejecutando tests de matplotlib...$(NC)"
	$(PYTHON) -m pytest tests/test_matplotlib_analyzer.py tests/test_matplotlib_formats.py -v

test-unit: ## Ejecutar solo tests unitarios
	@echo "$(YELLOW)Ejecutando tests unitarios...$(NC)"
	$(PYTHON) -m pytest tests/ -v -m "unit"

test-integration: ## Ejecutar solo tests de integración
	@echo "$(YELLOW)Ejecutando tests de integración...$(NC)"
	$(PYTHON) -m pytest tests/ -v -m "integration"

test-parallel: ## Ejecutar tests en paralelo
	@echo "$(YELLOW)Ejecutando tests en paralelo...$(NC)"
	$(PYTHON) -m pytest tests/ -v -n auto

lint: ## Ejecutar linting
	@echo "$(YELLOW)Ejecutando linting...$(NC)"
	@echo "$(BLUE)Black...$(NC)"
	black --check --diff .
	@echo "$(BLUE)isort...$(NC)"
	isort --check-only --diff .
	@echo "$(BLUE)flake8...$(NC)"
	flake8 $(PACKAGE_NAME)/ tests/
	@echo "$(BLUE)mypy...$(NC)"
	mypy $(PACKAGE_NAME)/
	@echo "$(BLUE)bandit...$(NC)"
	bandit -r $(PACKAGE_NAME)/

format: ## Formatear código
	@echo "$(YELLOW)Formateando código...$(NC)"
	black .
	isort .

clean: ## Limpiar archivos generados
	@echo "$(YELLOW)Limpiando archivos generados...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: ## Construir paquete
	@echo "$(YELLOW)Construyendo paquete...$(NC)"
	$(PYTHON) -m build

build-check: ## Verificar paquete construido
	@echo "$(YELLOW)Verificando paquete...$(NC)"
	twine check dist/*

publish-test: ## Publicar en TestPyPI
	@echo "$(YELLOW)Publicando en TestPyPI...$(NC)"
	twine upload --repository testpypi dist/*

publish: ## Publicar en PyPI
	@echo "$(YELLOW)Publicando en PyPI...$(NC)"
	twine upload dist/*

docs: ## Generar documentación
	@echo "$(YELLOW)Generando documentación...$(NC)"
	cd docs && make html

docs-serve: ## Servir documentación localmente
	@echo "$(YELLOW)Sirviendo documentación en http://localhost:8000...$(NC)"
	cd docs/_build/html && python -m http.server 8000

version: ## Mostrar versión actual
	@echo "$(GREEN)Versión actual: $(VERSION)$(NC)"

version-bump: ## Incrementar versión (patch)
	@echo "$(YELLOW)Incrementando versión patch...$(NC)"
	bump2version patch

version-minor: ## Incrementar versión (minor)
	@echo "$(YELLOW)Incrementando versión minor...$(NC)"
	bump2version minor

version-major: ## Incrementar versión (major)
	@echo "$(YELLOW)Incrementando versión major...$(NC)"
	bump2version major

docker-build: ## Construir imagen Docker
	@echo "$(YELLOW)Construyendo imagen Docker...$(NC)"
	docker build -t $(PACKAGE_NAME):latest .

docker-test: ## Ejecutar tests en Docker
	@echo "$(YELLOW)Ejecutando tests en Docker...$(NC)"
	docker-compose --profile test up --build --abort-on-container-exit

docker-jupyter: ## Ejecutar Jupyter en Docker
	@echo "$(YELLOW)Ejecutando Jupyter en Docker...$(NC)"
	@echo "$(GREEN)Jupyter disponible en http://localhost:8888$(NC)"
	docker-compose --profile jupyter up --build

docker-docs: ## Ejecutar documentación en Docker
	@echo "$(YELLOW)Ejecutando documentación en Docker...$(NC)"
	@echo "$(GREEN)Documentación disponible en http://localhost:8000$(NC)"
	docker-compose --profile docs up --build

docker-lint: ## Ejecutar linting en Docker
	@echo "$(YELLOW)Ejecutando linting en Docker...$(NC)"
	docker-compose --profile lint up --build --abort-on-container-exit

docker-clean: ## Limpiar contenedores Docker
	@echo "$(YELLOW)Limpiando contenedores Docker...$(NC)"
	docker-compose down --volumes --remove-orphans
	docker system prune -f

check: ## Verificar instalación
	@echo "$(YELLOW)Verificando instalación...$(NC)"
	$(PYTHON) -c "import $(PACKAGE_NAME); print('✅ $(PACKAGE_NAME) instalado correctamente')"
	$(PYTHON) -c "from $(PACKAGE_NAME) import convert; print('✅ Función convert disponible')"

check-deps: ## Verificar dependencias
	@echo "$(YELLOW)Verificando dependencias...$(NC)"
	@echo "$(BLUE)Python:$(NC)" && $(PYTHON) --version
	@echo "$(BLUE)Pip:$(NC)" && $(PIP) --version
	@echo "$(BLUE)Numpy:$(NC)" && $(PYTHON) -c "import numpy; print(numpy.__version__)"
	@echo "$(BLUE)Pandas:$(NC)" && $(PYTHON) -c "import pandas; print(pandas.__version__)"
	@echo "$(BLUE)Matplotlib:$(NC)" && $(PYTHON) -c "import matplotlib; print(matplotlib.__version__)" 2>/dev/null || echo "No instalado"
	@echo "$(BLUE)Seaborn:$(NC)" && $(PYTHON) -c "import seaborn; print(seaborn.__version__)" 2>/dev/null || echo "No instalado"
	@echo "$(BLUE)Plotly:$(NC)" && $(PYTHON) -c "import plotly; print(plotly.__version__)" 2>/dev/null || echo "No instalado"

example: ## Ejecutar ejemplo básico
	@echo "$(YELLOW)Ejecutando ejemplo básico...$(NC)"
	$(PYTHON) example.py

example-advanced: ## Ejecutar ejemplo avanzado
	@echo "$(YELLOW)Ejecutando ejemplo avanzado...$(NC)"
	$(PYTHON) example_advanced_analysis.py

example-seaborn: ## Ejecutar ejemplo de seaborn
	@echo "$(YELLOW)Ejecutando ejemplo de seaborn...$(NC)"
	$(PYTHON) example_seaborn.py

pre-commit: ## Instalar pre-commit hooks
	@echo "$(YELLOW)Instalando pre-commit hooks...$(NC)"
	pre-commit install

pre-commit-run: ## Ejecutar pre-commit en todos los archivos
	@echo "$(YELLOW)Ejecutando pre-commit en todos los archivos...$(NC)"
	pre-commit run --all-files

security: ## Ejecutar análisis de seguridad
	@echo "$(YELLOW)Ejecutando análisis de seguridad...$(NC)"
	safety check
	bandit -r $(PACKAGE_NAME)/

full-check: ## Ejecutar todas las verificaciones
	@echo "$(YELLOW)Ejecutando todas las verificaciones...$(NC)"
	$(MAKE) check
	$(MAKE) check-deps
	$(MAKE) lint
	$(MAKE) test-cov
	$(MAKE) security

release: ## Preparar release (clean, test, build, check)
	@echo "$(YELLOW)Preparando release...$(NC)"
	$(MAKE) clean
	$(MAKE) install-dev
	$(MAKE) full-check
	$(MAKE) build
	$(MAKE) build-check
	@echo "$(GREEN)✅ Release listo para publicación$(NC)"
	@echo "$(BLUE)Para publicar en TestPyPI: make publish-test$(NC)"
	@echo "$(BLUE)Para publicar en PyPI: make publish$(NC)"

dev-setup: ## Configurar entorno de desarrollo completo
	@echo "$(YELLOW)Configurando entorno de desarrollo...$(NC)"
	$(MAKE) install-dev
	$(MAKE) pre-commit
	$(MAKE) check
	@echo "$(GREEN)✅ Entorno de desarrollo configurado$(NC)"

# Comandos específicos para CI/CD
ci-test: ## Comando para CI (tests sin cobertura)
	$(PYTHON) -m pytest tests/ -v --tb=short

ci-lint: ## Comando para CI (linting)
	black --check .
	isort --check-only .
	flake8 $(PACKAGE_NAME)/ tests/
	mypy $(PACKAGE_NAME)/

ci-build: ## Comando para CI (build)
	$(PYTHON) -m build
	twine check dist/*

# Comandos de utilidad
stats: ## Mostrar estadísticas del proyecto
	@echo "$(YELLOW)Estadísticas del proyecto:$(NC)"
	@echo "$(BLUE)Líneas de código Python:$(NC)" && find $(PACKAGE_NAME)/ -name "*.py" -exec wc -l {} + | tail -1
	@echo "$(BLUE)Líneas de tests:$(NC)" && find tests/ -name "*.py" -exec wc -l {} + | tail -1
	@echo "$(BLUE)Archivos Python:$(NC)" && find . -name "*.py" | wc -l
	@echo "$(BLUE)Archivos de test:$(NC)" && find tests/ -name "*.py" | wc -l

todo: ## Mostrar TODOs en el código
	@echo "$(YELLOW)Buscando TODOs en el código...$(NC)"
	@grep -r "TODO" $(PACKAGE_NAME)/ tests/ || echo "No se encontraron TODOs"

fixme: ## Mostrar FIXMEs en el código
	@echo "$(YELLOW)Buscando FIXMEs en el código...$(NC)"
	@grep -r "FIXME" $(PACKAGE_NAME)/ tests/ || echo "No se encontraron FIXMEs"

# Comandos de ayuda adicional
install-tools: ## Instalar herramientas de desarrollo
	@echo "$(YELLOW)Instalando herramientas de desarrollo...$(NC)"
	$(PIP) install black isort flake8 mypy bandit safety pre-commit bump2version

update-deps: ## Actualizar dependencias
	@echo "$(YELLOW)Actualizando dependencias...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install --upgrade -r requirements.txt
	$(PIP) install --upgrade -e ".[dev,all]"

# Comandos de backup y restore
backup: ## Crear backup del proyecto
	@echo "$(YELLOW)Creando backup del proyecto...$(NC)"
	tar -czf backup-$(shell date +%Y%m%d-%H%M%S).tar.gz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' .

restore: ## Restaurar desde backup (especificar archivo con BACKUP_FILE=archivo.tar.gz)
	@if [ -z "$(BACKUP_FILE)" ]; then echo "$(RED)Especifica el archivo de backup: make restore BACKUP_FILE=archivo.tar.gz$(NC)"; exit 1; fi
	@echo "$(YELLOW)Restaurando desde $(BACKUP_FILE)...$(NC)"
	tar -xzf $(BACKUP_FILE) 