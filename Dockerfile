FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for some Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir keeps the image size smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the default Streamlit port
EXPOSE 8501

# Configure Streamlit to run headless and listen on all interfaces
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Run the Streamlit app
CMD ["streamlit", "run", "app.py"]
