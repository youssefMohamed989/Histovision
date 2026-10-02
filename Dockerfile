FROM python:3.11-slim

# OpenSlide and OpenCV runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        openslide-tools \
        libgl1 \
        libglib2.0-0 \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
# runtime dependencies only (no dev extras)
RUN pip install --no-cache-dir -e .

COPY configs ./configs
COPY scripts ./scripts

# run as an unprivileged user
RUN useradd --create-home app && mkdir -p /app/data /app/results /app/checkpoints \
    && chown -R app:app /app
USER app

EXPOSE 8000

# The API has no authentication: publish the port on localhost only (see docker-compose.yml).
CMD ["uvicorn", "liver_histo_ai.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
