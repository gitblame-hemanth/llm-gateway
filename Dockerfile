FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

COPY src/ src/
COPY config/ config/

# ---------------------------------------------------------------------------
FROM python:3.11-slim

RUN groupadd -r gateway && useradd -r -g gateway -d /app -s /sbin/nologin gateway

WORKDIR /app

COPY --from=builder /install /usr/local
COPY --from=builder /build/src src/
COPY --from=builder /build/config config/

RUN chown -R gateway:gateway /app

USER gateway

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
