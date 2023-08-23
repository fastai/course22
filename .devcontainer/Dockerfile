FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Update system
RUN apt update \
    && apt upgrade -y

# Setup non-root user vscode
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID --shell /bin/bash -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# [Optional] Change the User now. Omit if you want to keep the default as root.
# USER $USERNAME

# install python and system tools
RUN apt install software-properties-common wget curl python3-dev python3-pip python3-wheel python3-setuptools -y
# RUN /usr/bin/python3 -m pip install --upgrade pip -y
    
   # RUN apt install -y python3 \
#     && apt install -y git \
#     && apt install -y pip \
#     && /usr/bin/python3 -m pip install ipykernel -U --user --force-reinstall
COPY requirements.txt /tmp/

RUN echo "(*) Installing python packages..." \
    && su - $USERNAME -c "pip install -r /tmp/requirements.txt" \
    && apt-get -y install --no-install-recommends graphviz

ENV PATH="/home/vscode/.local/bin:${PATH}"
# USER $USERNAME  

COPY welcome-message.txt /usr/local/etc/vscode-dev-containers/first-run-notice.txt
