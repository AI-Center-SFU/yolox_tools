# Используем базовый образ с поддержкой GPU
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Устанавливаем основные инструменты
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

# Устанавливаем pip
RUN python3 -m pip install --upgrade pip

# Указываем рабочую директорию
WORKDIR /workspace

# Копируем все файлы из текущей директории в контейнер
COPY . /workspace

# Клонируем YOLOX репозиторий (если требуется отдельно)
RUN git clone https://github.com/aicsfu/YOLOX.git /workspace/YOLOX

# Переходим в директорию YOLOX
WORKDIR /workspace/YOLOX

# Устанавливаем PyTorch (версия совместима с CUDA 11.8)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Проверяем содержимое директории для отладки
RUN ls -la /workspace/YOLOX

# Устанавливаем YOLOX как библиотеку
RUN pip install -v -e .

# Устанавливаем Jupyter Notebook
RUN pip install notebook

# Указываем текущую директорию
WORKDIR /workspace

# Команда по умолчанию для запуска Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]