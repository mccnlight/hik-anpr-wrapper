# Образ с Python + ffmpeg для работы снеговой камеры через FFmpegRTSPReader на Render
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -U pip
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 10000

# Render задаёт PORT; без него по умолчанию 10000
CMD ["bash", "-lc", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
