#!/bin/bash

# Exit on errors
set -e

# Step 1: Generate static documentation using Google-style docstrings
echo "Generating static docs to ./docs ..."
pdoc -d google -o docs src

# Step 2: Serve the documentation locally
echo "Starting local doc server at http://localhost:8080 ..."
pdoc src -n -h localhost -p 8080

