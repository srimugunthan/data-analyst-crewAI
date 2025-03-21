# # Use Python 3.11.9 as base image
# FROM python:3.11.9-slim

# # Set the working directory in the container
# WORKDIR /app

# # Copy the current directory contents into the container
# COPY . /app

# # Install dependencies from requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the .env file if it’s used for environment variables
# COPY .env .env

# # Run the application
# CMD ["python", "src/newproj/main.py"] 

FROM python:3.11.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt FIRST for caching
COPY requirements.txt /app/

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install poetry
COPY poetry.lock pyproject.toml /app/

# Project initialization:
RUN poetry install --no-root
RUN pip install ipykernel pandas numpy matplotlib seaborn statsmodels scipy scikit-learn

# Copy the rest of the application code
COPY . /app/

# Copy the .env file if it’s used for environment variables (place after application code)
COPY .env /app/.env

# Run the application
CMD ["python", "src/newproj/main.py"]