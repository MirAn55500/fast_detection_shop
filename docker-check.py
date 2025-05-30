#!/usr/bin/env python3
"""
Скрипт проверки Docker-установки Object Detection Service
"""

import os
import sys
import subprocess
import time
import webbrowser
from urllib.request import urlopen
from urllib.error import URLError

def check_docker_installed():
    """Проверяет, установлен ли Docker и Docker Compose"""
    try:
        subprocess.run(["docker", "--version"], check=True, stdout=subprocess.PIPE)
        print("✅ Docker установлен")
        
        subprocess.run(["docker-compose", "--version"], check=True, stdout=subprocess.PIPE)
        print("✅ Docker Compose установлен")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker или Docker Compose не установлен")
        print("Пожалуйста, установите Docker и Docker Compose: https://docs.docker.com/get-docker/")
        return False

def check_service_running():
    """Проверяет, запущен ли сервис Object Detection"""
    try:
        response = urlopen("http://localhost:8000")
        if response.status == 200:
            print("✅ Сервис Object Detection работает на http://localhost:8000")
            return True
        else:
            print(f"❌ Сервис отвечает с кодом: {response.status}")
            return False
    except URLError:
        print("❌ Сервис не отвечает на http://localhost:8000")
        return False

def start_service():
    """Запускает сервис через Docker Compose"""
    print("🔄 Запуск сервиса через Docker Compose...")
    try:
        subprocess.run(["docker-compose", "up", "-d"], check=True)
        print("✅ Сервис запущен")
        
        # Ждем, пока сервис полностью загрузится
        max_attempts = 10
        for i in range(max_attempts):
            print(f"⏳ Ожидание запуска сервиса ({i+1}/{max_attempts})...")
            time.sleep(3)
            if check_service_running():
                return True
        
        print("❌ Сервис не запустился в течение ожидаемого времени")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка при запуске сервиса: {e}")
        return False

def main():
    """Основная функция скрипта"""
    print("🔍 Проверка установки Object Detection Service в Docker")
    
    if not check_docker_installed():
        return 1
    
    if not check_service_running():
        print("🔄 Сервис не запущен, пробуем запустить...")
        if not start_service():
            return 1
    
    # Открываем браузер с адресом сервиса
    print("🌐 Открываем сервис в браузере...")
    webbrowser.open("http://localhost:8000")
    
    print("\n✅ Проверка завершена успешно!")
    print("📝 Документация по использованию находится в README.md")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 