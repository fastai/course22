# Specify NVIDIA Devel Image
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Update Ubuntu software
RUN echo "(*) Update Ubuntu22 packages..." \
    && apt update \
    && apt upgrade -y

# install python and system tools
RUN echo "(*) Installing python and system tools..." \
    && apt-get install software-properties-common wget curl python3-dev python3-pip python3-wheel python3-setuptools -y \
    && apt-get install htop zlib1g g++ freeglut3-dev libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev libfreeimage-dev -y \
    && apt-get install --no-install-recommends apt-utils -y \
    && apt-get -y install procps lsb-release \
    && /usr/bin/python3 -m pip install --upgrade pip

# Need to clean up packages or we may run out of space in the cache
RUN apt-get clean

# Install libcudnn
RUN apt-get install libcudnn8=8.9.0.*-1+cuda11.8 
RUN apt-get install libcudnn8-dev=8.9.0.*-1+cuda11.8 -y

# Install python packages with pip
COPY requirements.txt /tmp/
RUN echo "(*) Installing python packages with pip..." \
    && su - $USERNAME -c "pip install -r /tmp/requirements.txt" \
    && apt-get -y install --no-install-recommends graphviz \
    && rm /tmp/requirements.txt
    
# Add .local/bin to path
ENV PATH="/home/vscode/.local/bin:${PATH}"

# Add welcome message
COPY welcome-message.txt /usr/local/etc/vscode-dev-containers/first-run-notice.txt
