# Base image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy Django project files
COPY . .

# Expose Django default port
EXPOSE 8004

# Collect static files (optional, for production)
RUN python manage.py collectstatic --noinput

# Default command to run Django development server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8004"]
