FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install spaCy model for NLP Features
RUN python -m spacy download en_core_web_sm

# Copy project files
COPY . .

# Set PYTHONPATH to root so 'from src...' works correctly
ENV PYTHONPATH="/app:$PYTHONPATH"

# Default entrypoint for pipeline execution
ENTRYPOINT ["python", "-m", "src.pipeline.pipeline_orchestrator"]
CMD ["--mode", "train"]
