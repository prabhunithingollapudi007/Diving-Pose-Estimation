# Use an official Python runtime as a parent image
FROM python:3.8-slim

# # Install system dependencies (e.g., FFmpeg for video processing if run_pipeline uses it)
# # Consult your pipeline.py's dependencies. If it uses FFmpeg, this is critical.
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     ffmpeg \
#     && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies first
# This leverages Docker's build cache
COPY ./requirements.txt /app/requirements.txt
# RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Copy your application code, including pipeline.py
COPY ./main.py /app/main.py
COPY ./pipeline.py /app/pipeline.py

# Create directories for uploads and outputs inside the container
# These will be the mount points for volumes if you use them
RUN mkdir -p uploads
RUN mkdir -p outputs

COPY outputs /app/outputs

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]