FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive

# Set nnU-Net environment variables
ENV nnUNet_raw=/tmp/nnUNet_raw \
    nnUNet_preprocessed=/tmp/nnUNet_preprocessed \
    nnUNet_results=/tmp/nnUNet_results

# Create nnU-Net directories
RUN mkdir -p $nnUNet_raw $nnUNet_preprocessed $nnUNet_results

# Install system dependencies
RUN apt-get update && apt-get install -y git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY code/docker/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt

# Default entrypoint
ENTRYPOINT ["/bin/bash"]