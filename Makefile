.PHONY: build up down logs ps check test clean help

# Default target when just running 'make'
help:
	@echo "Makefile для управления Docker-контейнером сервиса детекции объектов"
	@echo ""
	@echo "Доступные команды:"
	@echo "  make build    - Собрать Docker-образ"
	@echo "  make up       - Запустить сервис"
	@echo "  make down     - Остановить сервис"
	@echo "  make logs     - Показать логи сервиса"
	@echo "  make ps       - Показать статус контейнера"
	@echo "  make check    - Запустить скрипт проверки Docker-установки"
	@echo "  make clean    - Удалить временные файлы и очистить кэш"
	@echo "  make help     - Показать эту справку"

# Build Docker image
build:
	docker-compose build

# Start service
up:
	docker-compose up -d
	@echo "Сервис запущен на http://localhost:8000"

# Stop service
down:
	docker-compose down

# Show logs
logs:
	docker-compose logs -f

# Show container status
ps:
	docker-compose ps

# Run Docker check script
check:
	python docker-check.py

# Clean temporary files
clean:
	@echo "Очистка временных файлов..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".DS_Store" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	@echo "Готово"

# Clean Docker volumes (use with caution)
clean-docker:
	@echo "ВНИМАНИЕ: Будут удалены все контейнеры и тома Docker связанные с этим проектом."
	@echo "Это приведет к потере всех загруженных моделей и результатов обработки."
	@read -p "Вы уверены? (y/n) " answer; \
	if [ "$$answer" = "y" ]; then \
		docker-compose down -v; \
		echo "Контейнеры и тома удалены"; \
	else \
		echo "Операция отменена"; \
	fi 