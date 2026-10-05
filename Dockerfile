# The drill as a container. Worth building even if you only ever run it on
# localhost: it pins the environment, and it makes hosting a config change
# rather than a rewrite.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt requirements-web.txt ./
RUN pip install --no-cache-dir -r requirements-web.txt

COPY morning_drill/ ./morning_drill/
COPY web/ ./web/
COPY scripts/ ./scripts/
COPY PROJECT.md ./

# State lives on a mounted volume, never in the image layer: a container
# filesystem resets on every deploy, and drill_records.json is what makes the
# curriculum adapt. Lose it and delayed retrieval silently starts over.
ENV DRILL_DATA_DIR=/data

# Anything the person at the keyboard cannot see is not localhost any more.
ENV DRILL_EXEC_MODE=subprocess \
    DRILL_EXEC_TIMEOUT=15 \
    DRILL_EXEC_MEMORY_MB=2048

RUN useradd --create-home --uid 10001 drill \
    && mkdir -p /data/logs \
    && chown -R drill:drill /data /app

USER drill

EXPOSE 8000

CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8000"]
