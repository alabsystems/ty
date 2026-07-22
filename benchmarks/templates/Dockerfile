# Copyright 2026 Andrew Yates.
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0
#
# Template Dockerfile for reproducible benchmark evaluation
#
# TEMPLATE USAGE:
# 1. Copy this file to your project's benchmarks/ directory
# 2. Replace WORKDIR and COPY paths for your project structure
# 3. Add project-specific dependencies in the "Install dependencies" section

# Base image - use specific tag for reproducibility
FROM python:3.11-slim-bookworm

# Metadata
LABEL maintainer="Andrew Yates <andrewyates.name@gmail.com>"
LABEL description="Reproducible benchmark evaluation container"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -s /bin/bash evaluator
WORKDIR /home/evaluator

# Copy project files (customize for your project)
COPY --chown=evaluator:evaluator . /home/evaluator/project

# Install dependencies (customize for your project)
WORKDIR /home/evaluator/project
RUN pip install --no-cache-dir -r requirements.txt 2>/dev/null || true

# Switch to non-root user
USER evaluator

# Default entrypoint runs evaluation script
ENTRYPOINT ["./benchmarks/run_eval.sh"]
CMD ["--help"]
