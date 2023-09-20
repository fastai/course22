FROM mcr.microsoft.com/devcontainers/python:0-3.10

COPY requirements.txt /tmp/

RUN echo "(*) Installing tools..." \
    && su - vscode -c "pip install -r /tmp/requirements.txt" \
    && apt-get update \
    && apt-get -y install --no-install-recommends graphviz

ENV PATH="/home/vscode/.local/bin:${PATH}"

COPY welcome-message.txt /usr/local/etc/vscode-dev-containers/first-run-notice.txt
