FROM tensorflow/tensorflow:2.18.0-gpu

# Встановлення системних залежностей, потрібних для OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

# Встановлення робочої директорії
WORKDIR /app

# Копіюємо тільки requirements.txt на ранньому етапі (для використання layer cache)
COPY requirements.txt .

# Встановлюємо Python-залежності
RUN pip install --no-cache-dir -r requirements.txt

# Копіюємо решту проєкту
COPY . .

# Вказуємо порт для документації
EXPOSE 8080

# Запуск FastAPI через uvicorn. Cloud Run передає порт через змінну середовища.
CMD ["python", "server.py"]

