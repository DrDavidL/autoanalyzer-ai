# app/Dockerfile

# Use Python 3.9 slim image as the base image
FROM python:3.11-slim

# Set the working directory to /my_team within the container
WORKDIR /auto_analyze

# Install curl for the health check
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file and install Python dependencies
COPY requirements.txt ./
RUN pip3 install -r requirements.txt

# Copy your main application code and additional files such as prompts.py
COPY main.py ./
COPY prompts.py ./
COPY markdown_to_docx.py ./

# If there are other files or directories to include, add them here
# COPY other_file.py ./
# COPY your_directory/ ./your_directory/
COPY data/ ./data/
# COPY images/ ./images/
# COPY output/ ./output/
COPY .streamlit/ ./.streamlit/
COPY explanations/ ./explanations/
COPY static/ ./static/


# Expose port 8501 for Streamlit
EXPOSE 8501

# Define a health check for the container using curl
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Set the entrypoint to run the Streamlit application
ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
