#!/bin/bash

echo "🚂 Complete Railway Optimization Workflow"
echo "=========================================="

# Check if required files exist
if [ ! -f "train_simulation_output_before.csv" ]; then
    echo "❌ Error: train_simulation_output_before.csv not found!"
    echo "Please ensure your baseline simulation data is available."
    exit 1
fi

if [ ! -f "advanced_optimizer.py" ]; then
    echo "❌ Error: advanced_optimizer.py not found!"
    exit 1
fi

echo "✅ Found required files"

# Step 1: Run optimization
echo ""
echo "📊 Step 1: Running Advanced Optimization..."
echo "-" * 50
python3 advanced_optimizer.py

if [ $? -ne 0 ]; then
    echo "❌ Optimization failed!"
    exit 1
fi

# Check if optimized file was created
if [ ! -f "train_simulation_output_after.csv" ]; then
    echo "❌ Error: Optimization did not generate output file!"
    exit 1
fi

echo "✅ Optimization completed successfully!"

# Step 2: Show optimization summary
echo ""
echo "📈 Step 2: Optimization Results Summary"
echo "---------------------------------------"
python3 -c "
import pandas as pd
from railway_dashboard import RailwayAnalytics

# Load data
df_before = pd.read_csv('train_simulation_output_before.csv')
df_after = pd.read_csv('train_simulation_output_after.csv')

# Process data
for df in [df_before, df_after]:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['delay_minutes'] = df['delay_minutes'].fillna(0)
    df['station'] = df['station'].fillna('Unknown')

analytics = RailwayAnalytics()
metrics_before = analytics.calculate_metrics(df_before)
metrics_after = analytics.calculate_metrics(df_after)

print('🎯 KEY IMPROVEMENTS:')
improvements = []

# Headway violations
hv_improvement = ((metrics_before['headway_violations'] - metrics_after['headway_violations']) / metrics_before['headway_violations'] * 100)
if hv_improvement > 0:
    improvements.append(f'  ✅ Headway violations reduced by {hv_improvement:.1f}%')

# Halted trains
if metrics_after['halted_pct'] < metrics_before['halted_pct']:
    improvements.append(f'  ✅ Halted trains reduced by {metrics_before[\"halted_pct\"] - metrics_after[\"halted_pct\"]:.1f}%')

# Delayed trains
if metrics_after['delayed_pct'] < metrics_before['delayed_pct']:
    delay_improvement = ((metrics_before['delayed_pct'] - metrics_after['delayed_pct']) / metrics_before['delayed_pct'] * 100)
    improvements.append(f'  ✅ Delayed trains improved by {delay_improvement:.1f}%')

# Rerouting
if metrics_after['rerouted_pct'] > metrics_before['rerouted_pct']:
    reroute_increase = metrics_after['rerouted_pct'] - metrics_before['rerouted_pct']
    improvements.append(f'  🔧 Dynamic rerouting increased by {reroute_increase:.1f}% (optimization in action)')

if improvements:
    for improvement in improvements:
        print(improvement)
else:
    print('  📊 Analysis complete - see dashboard for detailed comparison')

print('')
print('📊 Full metrics available in dashboard!')
"

# Step 3: Launch dashboard
echo ""
echo "🚀 Step 3: Launching Dashboard..."
echo "--------------------------------"
echo "Your optimized railway system is ready!"
echo ""
echo "📍 Dashboard will open at: http://localhost:8501"
echo "📁 Files ready for upload:"
echo "   - Baseline: train_simulation_output_before.csv"
echo "   - Optimized: train_simulation_output_after.csv"
echo ""
echo "🎨 Don't forget to:"
echo "   1. Toggle the theme (Light/Dark) in the sidebar"
echo "   2. Upload both files to see comparisons"
echo "   3. Explore all the optimization improvements!"
echo ""
echo "⏹️  Press Ctrl+C to stop the dashboard when done"
echo ""

# Launch Streamlit
streamlit run railway_dashboard.py

echo ""
echo "🎯 Workflow completed! Your railway optimization system is fully operational."