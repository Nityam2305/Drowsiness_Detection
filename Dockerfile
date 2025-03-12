# Use Python 3.11 as base image
FROM python:3.11

# Set working directory
WORKDIR /app

# Copy all files to container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Gradio's default port
EXPOSE 7860

# Run the application
CMD ["python", "drowsiness.py"]
