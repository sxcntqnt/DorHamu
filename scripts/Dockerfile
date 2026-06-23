FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    sumo \
    sumo-tools \
    libspatialindex-dev \
    curl \
    git \
    rustc \
    cargo \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install AStreet from the latest release

RUN git clone https://github.com/a-b-street/abstreet.git /abstreet \

    && cd /abstreet \

    && git checkout tags/v0.3.47 -b release-0.3.47 \

    && cargo build --release \

    && mv target/release/abstreet /usr/local/bin/ \

    && cd .. \

    && rm -rf /abstreet


# Install Cloverleaf

RUN git clone https://github.com/Refefer/cloverleaf.git /cloverleaf \

    && cd /cloverleaf \

    && cargo build --release \

    && cd .. \

    && rm -rf /cloverleaf

# Set environment variables
ENV SUMO_HOME=/usr/share/sumo
ENV PYTHONUNBUFFERED=1
ENV GOOGLE_CLOUD_PROJECT=my_project

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy specific directories and files
COPY configs/ /app/configs/
COPY steps/ /app/steps/
COPY pipelines/ /app/pipelines/
COPY run.py /app/run.py

# Expose port
EXPOSE 8080

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run application
CMD ["gunicorn", "--workers=4", "--threads=2", "--bind=0.0.0.0:8080", "--timeout=120", "steps.evaluate_and_deploy_api:app"]
