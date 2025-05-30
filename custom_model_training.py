import os
import subprocess

# Настройки для дообучения
CUSTOM_DATASET_DIR = 'custom_dataset'
MODEL_DIR = 'models'
CONFIG_FILE = os.path.join(MODEL_DIR, 'yolov4-tiny-custom.cfg')
PRETRAINED_WEIGHTS = os.path.join(MODEL_DIR, 'yolov4-tiny.weights')
CUSTOM_WEIGHTS = os.path.join(MODEL_DIR, 'yolov4-tiny-custom.weights')
DATA_FILE = os.path.join(MODEL_DIR, 'custom.data')
NAMES_FILE = os.path.join(MODEL_DIR, 'custom.names')
TRAIN_LIST = os.path.join(MODEL_DIR, 'train.txt')
TEST_LIST = os.path.join(MODEL_DIR, 'test.txt')

# Создание необходимых папок
os.makedirs(MODEL_DIR, exist_ok=True)

# Создание файла custom.names с классами
def create_names_file():
    with open(NAMES_FILE, 'w') as f:
        f.write('person\n')
        f.write('pallet\n')
        f.write('dome\n')
    print(f'Создан файл {NAMES_FILE} с классами.')

# Создание файла custom.data
def create_data_file():
    with open(DATA_FILE, 'w') as f:
        f.write('classes = 3\n')
        f.write(f'train = {TRAIN_LIST}\n')
        f.write(f'valid = {TEST_LIST}\n')
        f.write(f'names = {NAMES_FILE}\n')
        f.write('backup = backup/\n')
    print(f'Создан файл {DATA_FILE} с настройками данных.')

# Создание списков train.txt и test.txt
def create_train_test_lists():
    images = [f for f in os.listdir(os.path.join(CUSTOM_DATASET_DIR, 'images')) if f.endswith('.jpg')]
    total_images = len(images)
    train_count = int(total_images * 0.8)
    train_images = images[:train_count]
    test_images = images[train_count:]
    
    with open(TRAIN_LIST, 'w') as f:
        for img in train_images:
            f.write(f'{os.path.join(CUSTOM_DATASET_DIR, "images", img)}\n')
    with open(TEST_LIST, 'w') as f:
        for img in test_images:
            f.write(f'{os.path.join(CUSTOM_DATASET_DIR, "images", img)}\n')
    print(f'Создано {len(train_images)} изображений для обучения и {len(test_images)} для валидации.')

# Настройка конфигурационного файла YOLOv4-tiny
def setup_config_file():
    base_config = os.path.join(MODEL_DIR, 'yolov4-tiny.cfg')
    if not os.path.exists(base_config):
        raise FileNotFoundError(f'Файл конфигурации {base_config} не найден. Убедитесь, что он загружен.')
    
    with open(base_config, 'r') as f:
        config_content = f.read()
    
    # Изменение параметров для пользовательского набора данных
    config_content = config_content.replace('classes=80', 'classes=3')
    config_content = config_content.replace('filters=255', 'filters=24')  # (classes + 5)*3
    
    with open(CONFIG_FILE, 'w') as f:
        f.write(config_content)
    print(f'Создан пользовательский файл конфигурации {CONFIG_FILE}.')

# Запуск дообучения с помощью Darknet
def train_model():
    darknet_path = './darknet'  # Укажите путь к исполняемому файлу Darknet
    if not os.path.exists(darknet_path):
        raise FileNotFoundError(f'Darknet не найден по пути {darknet_path}. Убедитесь, что он установлен.')
    
    command = [
        darknet_path,
        'detector',
        'train',
        DATA_FILE,
        CONFIG_FILE,
        PRETRAINED_WEIGHTS,
        '-dont_show',  # Не показывать графики
        '-map'  # Вычислять mAP
    ]
    
    print(f'Запускаю дообучение с командой: {" ".join(command)}')
    process = subprocess.run(command, shell=False)
    if process.returncode != 0:
        print('Ошибка при дообучении модели. Проверьте вывод для деталей.')
    else:
        print('Дообучение завершено.')

def main():
    print('Настройка дообучения модели YOLOv4-tiny на пользовательском наборе данных...')
    create_names_file()
    create_data_file()
    create_train_test_lists()
    setup_config_file()
    print('Все файлы настроены. Начинаю дообучение...')
    train_model()

if __name__ == '__main__':
    main() 