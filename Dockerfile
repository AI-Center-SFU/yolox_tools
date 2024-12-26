# Используем базовый образ с поддержкой GPU
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Устанавливаем основные инструменты
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Обновляем pip
RUN python3 -m pip install --upgrade pip

# Указываем рабочую директорию
WORKDIR /workspace

# Копируем файлы из текущей директории в контейнер
COPY . /workspace

# Клонируем YOLOX (пример, если нужно отдельное клонирование)
RUN git clone https://github.com/aicsfu/YOLOX.git /workspace/YOLOX

# Переходим в директорию YOLOX
WORKDIR /workspace/YOLOX

# Устанавливаем PyTorch (версия совместима с CUDA 11.8)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Устанавливаем YOLOX как библиотеку
RUN pip install -v -e .

# Устанавливаем Jupyter Lab
RUN pip install jupyterlab

# Возвращаемся в рабочую директорию /workspace
WORKDIR /workspace

# Команда по умолчанию (запуск Jupyter Lab на 0.0.0.0:8888)
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--ServerApp.allow_root=True", "--ServerApp.allow_remote_access=True", "--ServerApp.token=", "--ServerApp.password="]

