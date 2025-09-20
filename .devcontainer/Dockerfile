# Image de base avec support GPU + Python + Node.js
FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    vim \
    jq \
    libgl1 \
    git-lfs \
    libglib2.0-0 \
    curl git python3 python3-pip python3-venv build-essential rsync \
    && ln -s /usr/bin/python3 /usr/bin/python

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Set SHELL environment variable for pnpm installer
ENV SHELL=/bin/bash
ENV UV_HTTP_TIMEOUT=120

# Installe pipenv ou autre gestionnaire si tu veux
RUN pip install --upgrade pip
RUN git config --global user.name "Benjamin Marchand"
RUN git config --global user.email "ben.marchand@free.fr"
RUN git config --global core.editor "vim"