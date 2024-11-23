FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV APP_HOME /app
ENV PORT 8080

# Set working directory
WORKDIR $APP_HOME

# Copy requirements.txt and install dependencies
COPY requirements.txt ./
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy application code
COPY . ./

# Create upload folder
RUN mkdir -p static/uploads

# Expose port
EXPOSE $PORT

# Run Flask app using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "--timeout", "0", "app:app"]
