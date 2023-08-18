# base image
FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

# avoid warnings by switching to noninteractive
ARG DEBIAN_FRONTEND=noninteractive

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir tiktokx

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run main.py when the container launches
CMD ["python", "main.py"]