FROM python:3.10-slim

# Install system dependencies required by OpenCV and video processing
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app/

# Install the Python package and its dependencies
RUN pip install --no-cache-dir -e .

# Set the entrypoint to the CLI
ENTRYPOINT ["python", "-m", "bedwatch.cli"]
