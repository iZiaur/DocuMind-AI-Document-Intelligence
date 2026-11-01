# Multi-stage / Production-ready Backend Dockerfile for DocuMind AI
FROM python:3.11-slim

# Prevent Python from writing bytecode and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies (curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root application user
RUN groupadd -g 10001 documind && \
    useradd -u 10001 -g documind -m -s /bin/bash documind

# Install Python requirements in a dedicated layer
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy application code and Alembic configuration
COPY backend /app/backend
COPY alembic.ini /app/alembic.ini

# Create persistent storage directory with non-root ownership
RUN mkdir -p /app/storage/documents && \
    chown -R documind:documind /app

# Expose backend API port
EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Run as non-root user
USER documind

# Production Uvicorn entrypoint
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
