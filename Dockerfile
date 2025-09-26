# Use the official slim Python image
FROM python:3.12-slim

# Install OS-level build tools (if you ever need to compile wheels)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy your app’s requirements and install them
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy the rest of your code in
COPY . .

# Tell Cloud Run which port to listen on
ENV PORT 8080
EXPOSE 8080

# Run the Flask app with Gunicorn, allowing 120s for workers to boot
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "600", "main:app"]
