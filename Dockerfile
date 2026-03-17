FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

COPY app/requirements.txt /build/requirements.txt

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --upgrade pip \
    && python -m pip wheel --wheel-dir /wheels -r /build/requirements.txt

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_DATA_DIR=/data

WORKDIR /opt/app

COPY app/requirements.txt /opt/app/requirements.txt
COPY --from=builder /wheels /wheels

RUN python -m pip install --upgrade pip \
    && python -m pip install /wheels/* \
    && rm -rf /wheels \
    && python -c "import imageio_ffmpeg,pathlib; e=imageio_ffmpeg.get_ffmpeg_exe(); t=pathlib.Path('/usr/local/bin/ffmpeg'); (t.exists() or t.is_symlink()) and t.unlink(); t.symlink_to(e); print(f'Linked ffmpeg -> {e}')"

COPY app/main.py /opt/app/main.py

EXPOSE 8080

CMD ["uvicorn", "main:app", "--app-dir", "/opt/app", "--host", "0.0.0.0", "--port", "8080"]
