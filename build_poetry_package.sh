#!/bin/bash
# Script to build a Poetry package for PyRIT with SQLAlchemy 1.4 compatibility

set -e  # Exit on error

# Save the original pyproject.toml
echo "Backing up original pyproject.toml..."
cp pyproject.toml pyproject.toml.original

# Use the Poetry version of pyproject.toml
echo "Using Poetry configuration..."
cp pyproject.poetry.toml pyproject.toml

# Use the SQLAlchemy 1.4 README
echo "Using SQLAlchemy 1.4 README..."
cp README.sqlalchemy14.md README.md.sqlalchemy14
cp README.md README.md.original
cp README.sqlalchemy14.md README.md

# Build the package
echo "Building package with Poetry..."
poetry build

# Restore original files
echo "Restoring original files..."
mv pyproject.toml.original pyproject.toml
mv README.md.original README.md

echo "Package built successfully! Check the 'dist' directory for the package files."
