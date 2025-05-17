# Base image with Python and common tools
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Copy requirement files
COPY requirements.txt .

# Install dependencies (including system ones)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy application code
COPY . .

# Create logs directory
RUN mkdir -p logs
# Download required NLTK corpora
RUN python -m nltk.downloader words stopwords wordnet vader_lexicon omw-1.4

# Set default command
CMD ["python", "main.py", "--config=config.yaml"]
