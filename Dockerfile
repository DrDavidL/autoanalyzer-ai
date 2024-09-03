# Use Python 3.11 slim image as the base image
FROM python:3.10-slim

# Set the working directory to /auto_analyze within the container
WORKDIR /auto_analyze

# Install system dependencies and curl for the health check in one RUN command
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    gcc \
    g++ \
    cmake \
    libffi-dev \
    libssl-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file and install Python dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the main application code and additional necessary files
COPY main.py prompts.py markdown_to_docx.py ./
COPY data/ ./data/
COPY .streamlit/ ./.streamlit/
COPY explanations/ ./explanations/
COPY static/ ./static/

# Expose port 8501 for Streamlit
EXPOSE 8501

# Define a health check for the container using curl
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Set the entrypoint to run the Streamlit application
ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
