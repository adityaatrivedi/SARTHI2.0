#!/bin/bash

echo "🚂 Railway Simulation Analytics Dashboard Setup"
echo "=============================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    echo "Please install Python 3 and try again."
    exit 1
fi

# Check if pip is installed  
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is required but not installed."
    echo "Please install pip3 and try again."
    exit 1
fi

echo "✅ Python and pip found"

# Install requirements
echo "📦 Installing Python dependencies..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Check if transformed data exists, if not create it
if [ ! -f "train_simulation_output_before_transformed.csv" ] || [ ! -f "train_simulation_output_after_transformed.csv" ]; then
    echo "🔄 Transforming simulation data..."
    python3 data_transformer.py
    echo "✅ Data transformation completed"
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "To run the dashboard:"
echo "  streamlit run railway_dashboard.py"
echo ""
echo "Or use the run script:"
echo "  ./run_dashboard.sh"
echo ""
echo "The dashboard will open in your browser at http://localhost:8501"