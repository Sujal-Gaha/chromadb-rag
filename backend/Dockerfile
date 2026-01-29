FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Create data directories
RUN mkdir -p /home/appuser/chroma_data /home/appuser/uploads /home/appuser/logs

# Copy application code
COPY --chown=appuser:appuser app/ ./app/

# Set environment variables
ENV CHROMA_PERSIST_PATH=/home/appuser/chroma_data
ENV CHROMA_UPLOAD_DIR=/home/appuser/uploads

EXPOSE 4321

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "4321"]
