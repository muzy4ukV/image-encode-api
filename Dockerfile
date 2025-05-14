FROM python:3.12-slim

# Встановлення системних залежностей, потрібних для OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Встановлюємо робочу директорію
WORKDIR /usr/src/app

# Копіюємо всі файли проєкту до контейнера
COPY . .

# Встановлюємо залежності
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt


# Вказуємо порт для документації
EXPOSE 8080

# Запуск FastAPI через uvicorn. Cloud Run передає порт через змінну середовища.
CMD ["python", "server.py"]

