#!/bin/bash

# Make sure we're in the right directory
cd "$(dirname "$0")"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
  echo "Installing dependencies..."
  npm install
fi

# Run the frontend
echo "Starting frontend server at http://localhost:3000"
npm start 