FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for some Python packages
# curl is used by the HEALTHCHECK instruction below
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir keeps the image size smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Pre-create the FAISS index directory so the PVC mount point exists
# and the app can write to it on first run without permission errors
RUN mkdir -p /app/faiss_index

# Run as a non-root user — security best practice for K8s production workloads
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

# Expose the default Streamlit port and the Prometheus metrics port
EXPOSE 8501
EXPOSE 9091

# Configure Streamlit to run headless and listen on all interfaces
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Health check — hits the Prometheus metrics endpoint which is always up
# once the app starts; works for plain Docker runs without a K8s Operator
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:9091/metrics || exit 1

# Run the Streamlit app
CMD ["streamlit", "run", "app.py"]
