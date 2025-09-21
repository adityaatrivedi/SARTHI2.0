# 🚂 Railway Simulation Analytics Dashboard

A comprehensive Streamlit-based dashboard for comparative analysis of train schedule optimization with advanced metrics, visualizations, and audit trail logging.

## 🌟 Features

### 📊 Key Performance Indicators (KPIs)
- **Total Trains Simulated** - Count of unique trains in the system
- **Active Trains** - Trains currently in operation 
- **Delayed Trains (>5min)** - Percentage of trains with significant delays
- **Halted Trains** - Percentage of trains stopped due to conflicts
- **Rerouted Trains** - Percentage of trains that changed routes
- **Average Headway Distance** - Mean separation between consecutive trains
- **500m Rule Violations** - Total safety rule violations

### 📈 Comparative Visualizations
1. **KPI Cards** - Side-by-side before/after metrics with color-coded improvement indicators
2. **Metrics Comparison Chart** - Bar chart comparing key performance indicators
3. **Headway Distance Histogram** - Distribution analysis with safety rule overlay
4. **Train Movement Timeline** - Gantt-style charts showing train schedules
5. **Station Congestion Heatmap** - Hour-by-hour congestion analysis
6. **Platform Utilization** - Resource utilization comparison
7. **Track Overlay Visualization** - Real-time track layout with violation markers

### 🛤️ Track Safety Analysis
- **Red markers** - Trains violating 500m safety rule
- **Green markers** - Compliant train positions
- Interactive hover details with train IDs and positions

### 📋 Audit Trail & Reporting
- **Append-only logging** - Complete audit trail of all analysis runs
- **CSV Export** - Downloadable comparison reports
- **Automated insights** - AI-generated summary of improvements and concerns
- **Timestamp tracking** - Full analysis history

### 🎨 Theme System
- **Light/Dark theme toggle** - Switch between themes for optimal visibility
- **Enhanced KPI cards** - Gradient backgrounds with proper contrast
- **Improved readability** - Theme-specific colors for text and backgrounds
- **Visual indicators** - Current theme display in sidebar
- **Hover effects** - Interactive elements with smooth transitions

## 🚀 Quick Start

### Option 1: Automated Setup
```bash
chmod +x setup_dashboard.sh
./setup_dashboard.sh
./run_dashboard.sh
```

### Option 2: Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# No transformation needed - works with original format

# Run the dashboard
streamlit run railway_dashboard.py
```

## 📁 Data Format

### Expected CSV Structure
Your simulation files should contain these columns:
```
timestamp, train_id, train_type, line, position_m, speed_kmph, station, event, delay_minutes
```

### Sample Data Row
```csv
1900-01-01 00:00:00,PA-074,Passenger,single_up,466,56,,moving,131
```

### Data Format Details
- **timestamp**: Simulation time
- **train_id**: Unique train identifier (e.g., PA-074, SU-035)
- **train_type**: Type of train (Passenger, Express, Superfast, MEMU, Mail, Freight)
- **line**: Track line (single_up, single_down, central, etc.)
- **position_m**: Train position in meters
- **speed_kmph**: Train speed in km/h
- **station**: Station name (can be empty)
- **event**: Train status (moving, halted, rerouted, scheduled)
- **delay_minutes**: Delay in minutes

## 📊 Dashboard Sections

### 1. Theme & Data Upload
- **Theme toggle** - Light/Dark mode switcher at top of sidebar
- **Theme indicator** - Shows current theme (☀️ Light Mode / 🌙 Dark Mode)
- **Sidebar file uploaders** for before/after CSV files
- **Real-time validation** with error messaging
- **Sample data format** display for guidance

### 2. KPI Overview
- **Color-coded metrics** (Green=improved, Red=worsened, Gray=unchanged)
- **Percentage changes** with directional arrows
- **Side-by-side comparison** layout

### 3. Comparative Charts
- **Interactive Plotly visualizations**
- **Before/after overlays** for easy comparison
- **Drill-down capabilities** with hover details

### 4. Track Visualization
- **Real-time track layout** showing train positions
- **Safety compliance markers** (red/green system)
- **Interactive train details** on hover

### 5. Export & Reporting
- **One-click CSV downloads** for reports
- **Audit trail access** with complete history
- **Automated insights** with actionable recommendations

## 📈 Metrics Calculation

### Delay Analysis
- Trains with >5 minutes delay from scheduled time
- Calculated as percentage of total trains
- Lower percentages indicate better performance

### Headway Compliance
- Distance between consecutive trains measured in meters
- Safety rule: minimum 500m separation required
- Violations tracked and highlighted in red

### Congestion Analysis
- Station-wise train count per hour
- Heatmap visualization showing peak times
- Platform utilization across different periods

### Route Optimization
- Track rerouting based on line changes and rerouted events
- Percentage of trains that changed their original line/route
- Uses 'rerouted' event type and line switching analysis

## 🔍 Key Insights Generation

The dashboard automatically generates intelligent insights such as:
- "Optimization reduced headway violations by 45% and improved average delays by 2.3 minutes"
- "Congestion at Hoshangabad remains high due to limited platforms"
- Recommendations for further improvements

## 📝 Audit Trail

Every analysis run is logged with:
- **Timestamp** of analysis
- **Source files** used
- **Before/after metrics** with percentage changes
- **Improvement direction** (Improved/Worsened/Unchanged)
- **Export capability** for compliance tracking

## 🛠️ Technical Requirements

### Dependencies
- Python 3.8+
- Streamlit 1.28+
- Pandas 2.0+
- Plotly 5.15+
- NumPy 1.24+
- Matplotlib 3.7+
- Seaborn 0.12+

## 📂 File Structure

```
SARTHI2.0/
├── railway_dashboard.py          # Main dashboard application
├── data_transformer.py           # Data format converter
├── requirements.txt               # Python dependencies  
├── setup_dashboard.sh            # Automated setup script
├── run_dashboard.sh              # Dashboard launcher
├── README.md                     # This documentation
├── train_simulation_output_before.csv      # Sample baseline data
├── train_simulation_output_after.csv       # Sample optimized data
├── comparison_report.csv         # Generated comparison report
└── audit_trail.log              # Analysis history log
```

## 🚦 Usage Guidelines

### Data Quality
- Ensure timestamp consistency across both files
- Validate train IDs are consistent between before/after datasets
- Check for missing or null values in critical columns

### Performance Tips
- For large datasets (>100k rows), consider sampling for initial analysis
- Use date filtering to focus on specific time periods
- Close unused browser tabs to free up memory

### Best Practices
- Always backup original data files before transformation
- Review audit trail regularly for compliance tracking
- Export reports after each significant analysis
- Use meaningful file names with timestamps

## 🐛 Troubleshooting

### Common Issues
1. **Import Errors**: Run `pip install -r requirements.txt`
2. **Data Format Issues**: Use `data_transformer.py` to convert data
3. **Memory Issues**: Reduce dataset size or increase system RAM
4. **Port Conflicts**: Change port in run script if 8501 is busy

### Support
For technical issues or feature requests, check the audit trail logs for error details and ensure all dependencies are properly installed.

## 🎯 Future Enhancements
- Real-time data streaming integration
- Machine learning-based delay prediction
- Advanced optimization algorithms
- Multi-language support
- Mobile-responsive design
- API integration capabilities

