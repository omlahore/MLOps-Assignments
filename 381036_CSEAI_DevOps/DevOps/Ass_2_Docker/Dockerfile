# Use the official Python image from the Docker Hub
FROM python:3.11.5

# Set the working directory in the container
WORKDIR /Docker_Assignment
# Copy the requirements.txt file to the container
COPY requirement.txt .

# Install the required packages
RUN pip install -r requirement.txt

# Copy the rest of the application code to the container
COPY . .

# Expose the port on which the app runs
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
