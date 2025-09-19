#!/bin/bash

echo "🚂 Starting Railway Simulation Analytics Dashboard..."
echo "===================================================="

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit is not installed."
    echo "Please run ./setup_dashboard.sh first"
    exit 1
fi

# Check if the main app file exists
if [ ! -f "railway_dashboard.py" ]; then
    echo "❌ railway_dashboard.py not found"
    exit 1
fi

echo "🚀 Launching dashboard..."
echo "📍 URL: http://localhost:8501"
echo "⏹️  Press Ctrl+C to stop the dashboard"
echo ""

# Run the Streamlit app
streamlit run railway_dashboard.py --server.port 8501 --server.address localhost