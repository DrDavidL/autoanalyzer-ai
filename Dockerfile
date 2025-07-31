# Use Python 3.11 slim image with UV pre-installed
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# Set the working directory 
WORKDIR /auto_analyze

# Install curl for the health check
RUN apt-get update --allow-releaseinfo-change && \
    apt-get install -y --no-install-recommends --allow-unauthenticated ca-certificates debian-archive-keyring && \
    apt-get update --allow-releaseinfo-change && \
    apt-get install -y curl build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set UV to use the system Python and copy mode for linking
ENV UV_SYSTEM_PYTHON=1
ENV UV_LINK_MODE=copy

# Copy the requirements.txt file and install Python dependencies
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt

# Copy the main application code and additional necessary files
COPY main.py prompts.py markdown_to_docx.py data_processing.py llm_integration.py ml.py pca.py plotting.py stats.py ui.py utils.py datagov_integration.py./
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
