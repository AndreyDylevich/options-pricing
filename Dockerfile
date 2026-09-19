# Образ приложения. База поднимается отдельным сервисом, см. compose.yml.

FROM python:3.12-slim

# PYTHONUNBUFFERED: без него вывод приложения висит в буфере и в
# `docker compose up` ничего не видно, пока буфер не заполнится.
# PYTHONDONTWRITEBYTECODE: .pyc в образе не нужны, он одноразовый.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Зависимости отдельным слоем и до кода: пересобирать их при каждой
# правке исходников незачем, а numpy со scipy ставятся долго.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Работаем не от root: внутри контейнера это лишние права.
RUN useradd --create-home pricing && chown -R pricing:pricing /app
USER pricing

EXPOSE 8000

# Миграции накатываются при каждом запуске: на чистой базе они создают
# схему, на уже накатанной не делают ничего. Так контейнер поднимается
# одной командой, без отдельного шага установки.
#
# exec нужен, чтобы uvicorn стал главным процессом и получал сигналы
# остановки: иначе `docker compose down` ждал бы таймаута вместо того,
# чтобы завершиться сразу.
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn app.main:app --host 0.0.0.0 --port 8000"]
