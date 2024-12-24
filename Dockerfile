# Start with a base image that includes Python
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Set the default command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
