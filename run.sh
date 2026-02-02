#!/bin/bash

# RAG Assistant Startup Script

echo "========================================="
echo "RAG Assistant for Senior Citizens"
echo "========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv venv
    echo "Virtual environment created."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "WARNING: .env file not found!"
    echo "Please copy .env.example to .env and add your API keys."
    exit 1
fi

# Install/update dependencies
echo "Checking dependencies..."
pip install -q -r requirements.txt

# Check if dataset folder exists
if [ ! -d "patientdata" ]; then
    echo "Creating patientdata folder..."
    mkdir -p patientdata
    echo "Please add your patient data text files to the patientdata/ folder."
fi

# Check if there are files in dataset
file_count=$(ls -1 patientdata/*.txt 2>/dev/null | wc -l)
if [ $file_count -eq 0 ]; then
    echo "WARNING: No .txt files found in patientdata/ folder!"
    echo "The application may not work properly without data."
fi

echo ""
echo "Starting application..."
echo "Access the app at: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop the server"
echo "========================================="

# Run the application
python app.py