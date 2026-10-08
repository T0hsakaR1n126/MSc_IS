#!/bin/bash
set -e

ROOT_DIR=$(pwd)
SRC_DIR="$ROOT_DIR/MonetDB-11.53.15"
BUILD_DIR="$SRC_DIR/build"
INSTALL_DIR="$ROOT_DIR/install"

echo "Cleaning..."
rm -rf "$INSTALL_DIR"
rm -rf "$BUILD_DIR"

echo "Configuring CMake..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" ..

echo "Building..."
cmake --build . -j$(nproc)
cmake --build . --target install

echo "Done! Installed to: $INSTALL_DIR"
