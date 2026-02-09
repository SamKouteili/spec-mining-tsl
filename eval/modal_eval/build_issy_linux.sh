#!/bin/bash
# Build an optimized issy binary for Linux using Docker
# This creates a binary that runs much faster than issy-static

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISSY_DIR="/Users/samkouteili/rose/tsl-f/issy"
OUTPUT_DIR="$SCRIPT_DIR"

echo "Building optimized issy binary for Linux..."
echo "Source: $ISSY_DIR"
echo "Output: $OUTPUT_DIR/issy-linux-optimized"

# Create a Dockerfile for building (use script dir to avoid /tmp permission issues)
cat > "$SCRIPT_DIR/issy-build.dockerfile" << 'EOF'
FROM haskell:9.6-slim

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    tar \
    gzip \
    pkg-config \
    libgmp-dev \
    zlib1g-dev \
    libncurses-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Spot library
RUN cd /tmp && \
    wget -q https://www.lrde.epita.fr/dload/spot/spot-2.11.6.tar.gz && \
    tar xzf spot-2.11.6.tar.gz && \
    cd spot-2.11.6 && \
    ./configure --disable-python --prefix=/usr/local && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd / && rm -rf /tmp/spot*

WORKDIR /app/issy

# Copy source and build with optimizations
# Use --install-ghc to install the correct GHC version specified in stack.yaml
COPY . .
RUN stack build --install-ghc --ghc-options='-O2' --copy-bins --local-bin-path /app/bin

# The binary will be at /app/bin/issy
EOF

# Build using Docker (cross-compile for x86_64 Linux, which Modal uses)
echo "Running Docker build for linux/amd64 (this may take 15-20 minutes first time)..."
echo "If you see disk errors, try: docker system prune -a && restart Docker Desktop"
docker build --platform linux/amd64 -f "$SCRIPT_DIR/issy-build.dockerfile" -t issy-builder "$ISSY_DIR"

# Extract the binary
echo "Extracting binary..."
docker create --name issy-extract issy-builder
docker cp issy-extract:/app/bin/issy "$OUTPUT_DIR/issy-linux-optimized"
docker rm issy-extract

# Make executable
chmod +x "$OUTPUT_DIR/issy-linux-optimized"

# Cleanup Dockerfile
rm -f "$SCRIPT_DIR/issy-build.dockerfile"

echo ""
echo "Done! Built: $OUTPUT_DIR/issy-linux-optimized"
echo ""
echo "File info:"
file "$OUTPUT_DIR/issy-linux-optimized"
ls -lh "$OUTPUT_DIR/issy-linux-optimized"
echo ""
echo "To use this binary, update synthesis_app.py to use ISSY_PATH = 'issy-linux-optimized'"
