FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive

# Set nnU-Net environment variables
ENV nnUNet_raw=/tmp/nnUNet_raw \
    nnUNet_preprocessed=/tmp/nnUNet_preprocessed \
    nnUNet_results=/tmp/nnUNet_results

# Create nnU-Net directories
RUN mkdir -p $nnUNet_raw $nnUNet_preprocessed $nnUNet_results

# Update package lists and install required packages
RUN apt-get update && apt-get install -y \
        git \
        openssh-server \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt into the image
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

# For (VS code) ssh connection 
RUN mkdir /var/run/sshd

# Default entrypoint
ENTRYPOINT ["/bin/bash"]