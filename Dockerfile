# Use an official Python runtime as a base image
FROM python:3.11.7

# Set the working directory inside the container
WORKDIR /app

# Copy all contents from the local machine to the container
COPY . /app

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Streamlit port
EXPOSE 8601

# Default command to run the Streamlit app
CMD ["streamlit", "run", "BioSimilarity_interface.py", "--server.port=8601", "--server.address=0.0.0.0", "--server.maxUploadSize=1000", "--server.maxMessageSize=500"]

