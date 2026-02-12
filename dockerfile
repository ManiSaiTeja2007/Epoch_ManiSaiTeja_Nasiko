# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/src 

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    --no-install-recommends \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code - NOTE: now copying to /app/src
COPY ./src /app/src
COPY .env.example /app/.env

# Create non-root user
RUN addgroup --system --gid 1001 appuser && \
    adduser --system --uid 1001 --gid 1001 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose port
EXPOSE 8000

# Run the application - NOTE: import path changed
CMD ["uvicorn", "src.app.__main__:app", "--host", "0.0.0.0", "--port", "8000"]