# Use Python 3.8.12-slim as the base image
FROM python:3.8.12-slim

# Install pipenv for managing dependencies
RUN pip install pipenv

# Set the working directory in the container
WORKDIR /app

# Copy Pipfile and Pipfile.lock to the container
COPY ["Pipfile", "Pipfile.lock", "./"]

# Install dependencies specified in Pipfile.lock
RUN pipenv install --system --deploy
RUN pip install werkzeug

# Ensure scikit-learn version is set to 1.5.1 to match the Pipfile requirement
# This avoids potential issues with version mismatches.
RUN pip install scikit-learn==1.3.2


# Copy application files to the container
COPY ["predict.py", "model_C=1.0.bin", "./"]

# Expose the port that the app will run on
EXPOSE 9696

# Run the application with Gunicorn
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
