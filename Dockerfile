# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy backend requirements first for caching
COPY project-backend/requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend code
COPY project-backend /app

# Create a non-root user (required by Hugging Face)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Expose the port Hugging Face expects
EXPOSE 7860

# Start command
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--timeout", "120"]
