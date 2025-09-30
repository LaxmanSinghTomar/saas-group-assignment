# FAQ Assistant Dockerfile
FROM python:3.11-slim

# Install uv and curl
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files first (for layer caching)
COPY pyproject.toml uv.lock ./

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/processed config

# Install dependencies (uv will create venv in container)
# Note: Not using --frozen to allow uv to regenerate lockfile with container paths
RUN uv sync --no-dev

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application directly with Python from venv
CMD ["/app/.venv/bin/python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

