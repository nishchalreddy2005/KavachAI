# Use Python 3.11 slim image
FROM python:3.11-slim

# Create a non-root user setup first
RUN useradd -m -u 1000 user

# Set environment variables
ENV USER=user
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH

# Switch to the new user
USER user

# Set working directory to user home
WORKDIR $HOME/app

# Copy requirements
COPY --chown=user project-backend/requirements.txt requirements.txt

# Install dependencies (as user, into ~/.local)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend code
COPY --chown=user project-backend .

# Expose the port Hugging Face expects
EXPOSE 7860

# Start command
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--timeout", "120"]
