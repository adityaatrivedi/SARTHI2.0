import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
import os
from io import BytesIO
import base64

# Configure page
st.set_page_config(
    page_title="Railway Simulation Analytics Dashboard",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="expanded"
)

def apply_theme_css(dark_theme=False):
    """Apply theme-specific CSS styles"""
    if dark_theme:
        # Dark theme styles
        css_styles = """
        <style>
        .metric-card {
            background: linear-gradient(135deg, #1e2530 0%, #2a3441 100%);
            color: #ffffff;
            padding: 1.2rem;
            border-radius: 0.8rem;
            border-left: 6px solid #ff6b6b;
            margin-bottom: 1rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
            border: 1px solid #3d4852;
            transition: all 0.3s ease;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.5);
        }
        .metric-improved {
            border-left-color: #51cf66 !important;
            background: linear-gradient(135deg, #1a2f1a 0%, #2d4a2d 100%);
        }
        .metric-worsened {
            border-left-color: #ff6b6b !important;
            background: linear-gradient(135deg, #2f1a1a 0%, #4a2d2d 100%);
        }
        .metric-unchanged {
            border-left-color: #868e96 !important;
            background: linear-gradient(135deg, #262626 0%, #3d3d3d 100%);
        }
        .insight-box {
            background: linear-gradient(135deg, #1a2332 0%, #2d3d52 100%);
            color: #ffffff;
            padding: 1.2rem;
            border-radius: 0.8rem;
            border-left: 6px solid #2196f3;
            margin: 1rem 0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
            border: 1px solid #3d4852;
        }
        .metric-card h4 {
            color: #ffffff !important;
            margin-bottom: 0.8rem;
            font-weight: 600;
            text-shadow: 0 1px 2px rgba(0,0,0,0.3);
        }
        .metric-card p {
            color: #e0e0e0 !important;
            margin: 0.3rem 0;
            font-weight: 500;
        }
        .metric-card strong {
            color: #ffffff !important;
        }
        .insight-box h3, .insight-box h4 {
            color: #ffffff !important;
            margin-bottom: 0.5rem;
        }
        .insight-box p, .insight-box li {
            color: #e0e0e0 !important;
            line-height: 1.6;
        }
        .insight-box strong {
            color: #ffffff !important;
            font-weight: 600;
        }
        </style>
        """
    else:
        # Light theme styles
        css_styles = """
        <style>
        .metric-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            color: #212529;
            padding: 1.2rem;
            border-radius: 0.8rem;
            border-left: 6px solid #ff6b6b;
            margin-bottom: 1rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            border: 1px solid #dee2e6;
            transition: all 0.3s ease;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.2);
        }
        .metric-improved {
            border-left-color: #28a745 !important;
            background: linear-gradient(135deg, #f0fff4 0%, #e8f5e8 100%);
        }
        .metric-worsened {
            border-left-color: #dc3545 !important;
            background: linear-gradient(135deg, #fff5f5 0%, #fee6e6 100%);
        }
        .metric-unchanged {
            border-left-color: #6c757d !important;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        }
        .insight-box {
            background: linear-gradient(135deg, #f0f8ff 0%, #e3f2fd 100%);
            color: #212529;
            padding: 1.2rem;
            border-radius: 0.8rem;
            border-left: 6px solid #007bff;
            margin: 1rem 0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            border: 1px solid #bee5eb;
        }
        .metric-card h4 {
            color: #212529 !important;
            margin-bottom: 0.8rem;
            font-weight: 600;
            text-shadow: 0 1px 2px rgba(255,255,255,0.8);
        }
        .metric-card p {
            color: #495057 !important;
            margin: 0.3rem 0;
            font-weight: 500;
        }
        .metric-card strong {
            color: #212529 !important;
        }
        .insight-box h3, .insight-box h4 {
            color: #212529 !important;
            margin-bottom: 0.5rem;
        }
        .insight-box p, .insight-box li {
            color: #495057 !important;
            line-height: 1.6;
        }
        .insight-box strong {
            color: #212529 !important;
            font-weight: 600;
        }
        </style>
        """
    
    st.markdown(css_styles, unsafe_allow_html=True)

class RailwayAnalytics:
    def __init__(self):
        self.required_columns = [
            'timestamp', 'train_id', 'train_type', 'line', 'position_m', 
            'speed_kmph', 'station', 'event', 'delay_minutes'
        ]
        self.setup_logging()
    
    def setup_logging(self):
        """Setup audit trail logging"""
        logging.basicConfig(
            filename='audit_trail.log',
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def validate_data(self, df, filename):
        """Validate CSV data structure"""
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            st.error(f"Missing columns in {filename}: {missing_cols}")
            return False
        return True
    
    def load_data(self, uploaded_file):
        """Load and validate CSV data"""
        try:
            df = pd.read_csv(uploaded_file)
            if self.validate_data(df, uploaded_file.name):
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                # Fill NaN values in delay_minutes with 0
                df['delay_minutes'] = df['delay_minutes'].fillna(0)
                # Fill NaN values in station column
                df['station'] = df['station'].fillna('Unknown')
                return df
            return None
        except Exception as e:
            st.error(f"Error loading {uploaded_file.name}: {str(e)}")
            return None
    
    def calculate_metrics(self, df):
        """Calculate all comparative metrics for original data format"""
        metrics = {}
        total_trains = df['train_id'].nunique()

        # 1. Throughput: Trains that completed their journey
        finished_events = df[df['event'].str.contains('finish|arrive', case=False, na=False)]
        metrics['throughput'] = finished_events['train_id'].nunique()

        # 2. Average Delay: Average delay in minutes for all events
        metrics['avg_delay_minutes'] = df['delay_minutes'].mean()
        
        # 3. % of trains delayed by >5 min
        delayed_trains = df[df['delay_minutes'] > 5]['train_id'].nunique()
        metrics['delayed_pct'] = (delayed_trains / total_trains) * 100 if total_trains > 0 else 0
        
        # 4. % halted due to headway/conflicts (using event='halted')
        halted_records = len(df[df['event'] == 'halted'])
        total_records = len(df)
        metrics['halted_pct'] = (halted_records / total_records) * 100 if total_records > 0 else 0
        
        # 5. % rerouted (using event='rerouted' and line changes)
        rerouted_records = len(df[df['event'] == 'rerouted'])
        # Also count trains that changed lines
        train_line_changes = df.groupby('train_id')['line'].nunique()
        rerouted_trains = len(train_line_changes[train_line_changes > 1])
        total_rerouted = max(rerouted_records, rerouted_trains)
        metrics['rerouted_pct'] = (total_rerouted / len(df)) * 100 if len(df) > 0 else 0
        
        # 6. Average headway distance
        df_sorted = df.sort_values(['timestamp', 'position_m'])
        headway_distances = []
        
        for timestamp in df_sorted['timestamp'].unique():
            positions = df_sorted[df_sorted['timestamp'] == timestamp]['position_m'].sort_values()
            if len(positions) > 1:
                distances = positions.diff().dropna()
                headway_distances.extend(distances.tolist())
        
        metrics['avg_headway_distance'] = np.mean(headway_distances) if headway_distances else 0
        
        # 7. Total violations of 500m rule
        violations = sum(1 for d in headway_distances if d < 500)
        metrics['headway_violations'] = violations
        
        return metrics
    
    def create_kpi_cards(self, metrics_before, metrics_after):
        """Create KPI cards with improvement indicators"""
        st.subheader("📊 Key Performance Indicators")
        
        kpi_names = [
            ('Throughput', 'throughput'),
            ('Avg Delay (min)', 'avg_delay_minutes'),
            ('Delayed >5min (%)', 'delayed_pct'),
            ('Halted (%)', 'halted_pct'),
            ('Rerouted (%)', 'rerouted_pct'),
            ('Avg Headway (m)', 'avg_headway_distance'),
            ('500m Violations', 'headway_violations')
        ]
        
        cols = st.columns(len(kpi_names))
        
        for i, (name, key) in enumerate(kpi_names):
            with cols[i]:
                before_val = metrics_before[key]
                after_val = metrics_after[key]
                
                # Determine improvement direction
                if key in ['avg_delay_minutes', 'delayed_pct', 'halted_pct', 'rerouted_pct', 'headway_violations']:
                    # Lower is better
                    if after_val < before_val:
                        status = "improved"
                        color = "green"
                        arrow = "↓"
                    elif after_val > before_val:
                        status = "worsened"
                        color = "red"
                        arrow = "↑"
                    else:
                        status = "unchanged"
                        color = "gray"
                        arrow = "→"
                else:
                    # Higher is better (or neutral)
                    if after_val > before_val:
                        status = "improved"
                        color = "green"
                        arrow = "↑"
                    elif after_val < before_val:
                        status = "worsened"
                        color = "red"
                        arrow = "↓"
                    else:
                        status = "unchanged"
                        color = "gray"
                        arrow = "→"
                
                change_pct = ((after_val - before_val) / before_val * 100) if before_val != 0 else 0
                
                st.markdown(f"""
                <div class="metric-card metric-{status}">
                    <h4>{name}</h4>
                    <div style="display: flex; justify-content: space-between;">
                        <div>
                            <p><strong>Before:</strong> {before_val:.1f}</p>
                            <p><strong>After:</strong> {after_val:.1f}</p>
                        </div>
                        <div style="text-align: right; color: {color};">
                            <span style="font-size: 24px;">{arrow}</span>
                            <p>{change_pct:+.1f}%</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    def create_comparison_charts(self, df_before, df_after, metrics_before, metrics_after):
        """Create various comparison charts"""
        
        # 1. Metrics Comparison Bar Chart
        st.subheader("📈 Metrics Comparison")
        
        metrics_data = pd.DataFrame({
            'Metric': ['Delayed %', 'Halted %', 'Rerouted %', 'Headway Violations'],
            'Before': [metrics_before['delayed_pct'], metrics_before['halted_pct'], 
                      metrics_before['rerouted_pct'], metrics_before['headway_violations']],
            'After': [metrics_after['delayed_pct'], metrics_after['halted_pct'], 
                     metrics_after['rerouted_pct'], metrics_after['headway_violations']]
        })
        
        fig_metrics = px.bar(
            metrics_data.melt(id_vars='Metric', var_name='Period', value_name='Value'),
            x='Metric', y='Value', color='Period',
            title="Key Metrics Comparison",
            barmode='group'
        )
        st.plotly_chart(fig_metrics, use_container_width=True)
        
        # 2. Headway Distance Histogram
        st.subheader("📏 Headway Distance Distribution")
        
        # Calculate headway distances for both datasets
        def get_headway_distances(df):
            df_sorted = df.sort_values(['timestamp', 'position_m'])
            distances = []
            for timestamp in df_sorted['timestamp'].unique():
                positions = df_sorted[df_sorted['timestamp'] == timestamp]['position_m'].sort_values()
                if len(positions) > 1:
                    distances.extend(positions.diff().dropna().tolist())
            return distances
        
        headway_before = get_headway_distances(df_before)
        headway_after = get_headway_distances(df_after)
        
        fig_headway = go.Figure()
        fig_headway.add_trace(go.Histogram(x=headway_before, name="Before", opacity=0.7))
        fig_headway.add_trace(go.Histogram(x=headway_after, name="After", opacity=0.7))
        fig_headway.add_vline(x=500, line_dash="dash", line_color="red", 
                             annotation_text="500m Safety Rule")
        fig_headway.update_layout(title="Headway Distance Distribution", 
                                 xaxis_title="Distance (meters)", yaxis_title="Frequency")
        st.plotly_chart(fig_headway, use_container_width=True)
        
        # 3. Timeline Chart
        st.subheader("⏰ Train Movement Timeline")

        # compute sync range
        time_min = min(df_before["timestamp"].min(), df_after["timestamp"].min())
        time_max = max(df_before["timestamp"].max(), df_after["timestamp"].max())

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Before Optimization**")
            fig_timeline_before = self.create_timeline_chart(df_before, "Before", time_min, time_max)
            st.plotly_chart(fig_timeline_before, use_container_width=True)
        
        with col2:
            st.write("**After Optimization**")
            fig_timeline_after = self.create_timeline_chart(df_after, "After", time_min, time_max)
            st.plotly_chart(fig_timeline_after, use_container_width=True)
        
        # 4. Station Congestion Heatmap
        st.subheader("🏢 Station Congestion Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Before Optimization**")
            self.create_congestion_heatmap(df_before)
        
        with col2:
            st.write("**After Optimization**")
            self.create_congestion_heatmap(df_after)
        
        # 5. Line Utilization
        st.subheader("🚉 Line Utilization")
        
        self.create_platform_utilization(df_before, df_after)
    
    def create_timeline_chart(self, df, title, time_min, time_max):
        """Create timeline chart showing train events and positions"""
        fig = px.scatter(
            df,
            x="timestamp",
            y="train_id",
            color="event",
            hover_data=["event", "position_m", "speed_kmph", "line", "delay_minutes"],
            title=f"Train Movement Timeline - {title}",
        )

        fig.update_layout(
            xaxis=dict(
                title="Time",
                type="date",
                range=[time_min, time_max],
                tickformat="%H:%M",
                tickangle=-45,
                showgrid=True
            ),
            yaxis=dict(
                title="Train ID",
                categoryorder="category ascending"
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=50, r=20, t=40, b=40),
            height=600
        )

        return fig
    
    def create_congestion_heatmap(self, df):
        """Create station congestion heatmap based on station positions from stations.csv."""
        try:
            stations_df = pd.read_csv('stations.csv')
            stations_df = stations_df.sort_values('position_m').reset_index(drop=True)
        except FileNotFoundError:
            st.error("stations.csv not found. Cannot generate station congestion heatmap.")
            return

        if stations_df.empty:
            st.warning("stations.csv is empty. Cannot determine station boundaries.")
            return

        # Calculate midpoints between stations to define boundaries
        midpoints = (stations_df['position_m'] + stations_df['position_m'].shift(-1)) / 2
        midpoints = midpoints.dropna().tolist()
        
        station_names = stations_df['name'].tolist()

        def get_station_from_position(position):
            for i, midpoint in enumerate(midpoints):
                if position < midpoint:
                    return station_names[i]
            return station_names[-1] # Return last station for positions beyond the last midpoint

        df_copy = df.copy()
        df_copy['station_mapped'] = df_copy['position_m'].apply(get_station_from_position)
        
        # Create hourly bins
        df_copy['hour'] = df_copy['timestamp'].dt.hour
        
        # Group by station and hour
        congestion = df_copy.groupby(['station_mapped', 'hour']).size().unstack(fill_value=0)
        
        if congestion.empty:
            st.write("No congestion data to display")
            return
        
        # Ensure we have consistent hours (0-23)
        all_hours = range(24)
        for hour in all_hours:
            if hour not in congestion.columns:
                congestion[hour] = 0
        
        # Order stations as they appear on the line
        congestion = congestion.reindex(station_names, fill_value=0).dropna(how='all')
        
        # Sort columns
        congestion = congestion.reindex(sorted(congestion.columns), axis=1)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(congestion, annot=True, fmt='d', cmap='YlOrRd', ax=ax, 
                   cbar_kws={'label': 'Number of Trains'})
        ax.set_title('Station Congestion (Trains per Hour)')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Station')
        st.pyplot(fig)
        plt.close()
    
    def create_platform_utilization(self, df_before, df_after):
        """Create line utilization comparison"""
        line_before = df_before.groupby('line').size()
        line_after = df_after.groupby('line').size()
        
        # Combine data
        lines = set(line_before.index) | set(line_after.index)
        utilization_data = []
        
        for line in lines:
            utilization_data.append({
                'Line': line,
                'Before': line_before.get(line, 0),
                'After': line_after.get(line, 0)
            })
        
        util_df = pd.DataFrame(utilization_data)
        
        fig = px.bar(
            util_df.melt(id_vars='Line', var_name='Period', value_name='Usage'),
            x='Line', y='Usage', color='Period',
            title="Line Utilization Comparison",
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def create_track_overlay(self, df_before, df_after):
        """Create track overlay visualization"""
        st.subheader("🛤️ Track Overlay Visualization")
        
        col1, col2 = st.columns(2)
        
        for col, df, title in [(col1, df_before, "Before"), (col2, df_after, "After")]:
            with col:
                st.write(f"**{title} Optimization**")
                
                # Get latest timestamp data
                latest_time = df['timestamp'].max()
                latest_data = df[df['timestamp'] == latest_time].copy()
                
                # Calculate headway violations
                latest_data = latest_data.sort_values('position_m')
                latest_data['headway_violation'] = False
                
                for i in range(1, len(latest_data)):
                    distance = latest_data.iloc[i]['position_m'] - latest_data.iloc[i-1]['position_m']
                    if distance < 500:
                        latest_data.iloc[i, latest_data.columns.get_loc('headway_violation')] = True
                
                # Create track visualization
                fig = go.Figure()
                
                # Compliant trains (green)
                compliant = latest_data[~latest_data['headway_violation']]
                if not compliant.empty:
                    fig.add_trace(go.Scatter(
                        x=compliant['position_m'],
                        y=[1] * len(compliant),
                        mode='markers',
                        marker=dict(color='green', size=10, symbol='square'),
                        name='Compliant (>500m)',
                        text=compliant['train_id'],
                        hovertemplate='Train: %{text}<br>Position: %{x}m<br>Event: ' + compliant['event'].astype(str) + '<extra></extra>'
                    ))
                
                # Violating trains (red)
                violating = latest_data[latest_data['headway_violation']]
                if not violating.empty:
                    fig.add_trace(go.Scatter(
                        x=violating['position_m'],
                        y=[1] * len(violating),
                        mode='markers',
                        marker=dict(color='red', size=10, symbol='square'),
                        name='Violating (<500m)',
                        text=violating['train_id'],
                        hovertemplate='Train: %{text}<br>Position: %{x}m<br>Event: ' + violating['event'].astype(str) + '<extra></extra>'
                    ))
                
                fig.update_layout(
                    title=f"Track Layout - {title}",
                    xaxis_title="Position (meters)",
                    yaxis=dict(showticklabels=False, range=[0.5, 1.5]),
                    height=200,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def log_audit_trail(self, metrics_before, metrics_after, before_file, after_file):
        """Log audit trail entry"""
        improvements = {}
        
        metric_names = {
            'throughput': 'Throughput',
            'avg_delay_minutes': 'Avg Delay (min)',
            'delayed_pct': 'Delayed %',
            'halted_pct': 'Halted %',
            'rerouted_pct': 'Rerouted %',
            'avg_headway_distance': 'Avg Headway Distance',
            'headway_violations': 'Headway Violations'
        }
        
        for key, name in metric_names.items():
            before_val = metrics_before[key]
            after_val = metrics_after[key]
            
            if before_val != 0:
                change_pct = ((after_val - before_val) / before_val) * 100
            else:
                change_pct = 0
            
            if key in ['avg_delay_minutes', 'delayed_pct', 'halted_pct', 'rerouted_pct', 'headway_violations']:
                # Lower is better
                if after_val < before_val:
                    direction = "Improved"
                elif after_val > before_val:
                    direction = "Worsened"
                else:
                    direction = "Unchanged"
            else:
                # Higher is better or neutral
                if after_val > before_val:
                    direction = "Improved"
                elif after_val < before_val:
                    direction = "Worsened"
                else:
                    direction = "Unchanged"
            
            improvements[name] = {
                'before': before_val,
                'after': after_val,
                'change_pct': change_pct,
                'direction': direction
            }
        
        # Log entry
        log_entry = f"""
=== RAILWAY SIMULATION ANALYSIS ===
Dataset Before: {before_file}
Dataset After: {after_file}
Analysis Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

METRICS COMPARISON:
"""
        for name, data in improvements.items():
            log_entry += f"{name}: {data['before']:.2f} → {data['after']:.2f} ({data['change_pct']:+.1f}%) - {data['direction']}\n"
        
        log_entry += "\n" + "="*50 + "\n"
        
        logging.info(log_entry)
        
        return improvements
    
    def export_reports(self, metrics_before, metrics_after, improvements):
        """Export comparison report and provide download links"""
        
        # Create comparison report CSV
        report_data = []
        for metric, data in improvements.items():
            report_data.append({
                'Metric': metric,
                'Before': data['before'],
                'After': data['after'],
                'Change_%': data['change_pct'],
                'Direction': data['direction']
            })
        
        report_df = pd.DataFrame(report_data)
        
        # Save to CSV
        report_df.to_csv('comparison_report.csv', index=False)
        
        # Provide download link
        st.download_button(
            label="📥 Download Comparison Report (CSV)",
            data=report_df.to_csv(index=False),
            file_name=f"railway_comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # Display audit trail if it exists
        if os.path.exists('audit_trail.log'):
            with open('audit_trail.log', 'r') as f:
                log_content = f.read()
            
            st.download_button(
                label="📥 Download Audit Trail Log",
                data=log_content,
                file_name=f"audit_trail_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
                mime="text/plain"
            )
    
    def generate_insights(self, improvements):
        """Generate intelligent insights summary"""
        insights = []
        
        # Analyze key improvements
        delay_improvement = improvements.get('Delayed %', {})
        headway_improvement = improvements.get('Headway Violations', {})
        halted_improvement = improvements.get('Halted %', {})
        
        if delay_improvement.get('direction') == 'Improved':
            change = abs(delay_improvement['change_pct'])
            insights.append(f"✅ Delay performance improved by {change:.1f}%")
        
        if headway_improvement.get('direction') == 'Improved':
            reduction = headway_improvement['before'] - headway_improvement['after']
            pct_change = abs(headway_improvement['change_pct'])
            insights.append(f"✅ Headway violations reduced by {reduction:.0f} incidents ({pct_change:.1f}% improvement)")
        
        if halted_improvement.get('direction') == 'Improved':
            change = abs(halted_improvement['change_pct'])
            insights.append(f"✅ Train halts reduced by {change:.1f}%")
        
        # Look for areas still needing attention
        concerns = []
        for metric, data in improvements.items():
            if data['direction'] == 'Worsened' and abs(data['change_pct']) > 5:
                concerns.append(f"⚠️ {metric} worsened by {abs(data['change_pct']):.1f}%")
        
        # Combine insights
        if insights:
            summary = "**Optimization Results:** " + " | ".join(insights)
        else:
            summary = "**Analysis completed** - Mixed results observed across metrics"
        
        if concerns:
            summary += "\n\n**Areas for attention:** " + " | ".join(concerns)
        
        summary += "\n\n*Recommendation: Focus on high-congestion stations and continue monitoring headway compliance for sustained improvements.*"
        
        return summary

def main():
    """Main Streamlit application"""
    
    st.title("🚂 Railway Simulation Analytics Dashboard")
    st.markdown("**Comparative Analysis of Train Schedule Optimization**")
    
    analytics = RailwayAnalytics()
    
    # Theme toggle at top of sidebar
    st.sidebar.subheader("🎨 Theme Settings")
    dark_theme = st.sidebar.toggle("🌙 Dark Theme", value=False, help="Toggle between light and dark theme")
    
    # Show theme indicator
    theme_indicator = "🌙 Dark Mode" if dark_theme else "☀️ Light Mode"
    st.sidebar.markdown(f"**Current:** {theme_indicator}")
    
    # Apply theme-specific CSS
    apply_theme_css(dark_theme)
    
    st.sidebar.markdown("---")
    # Sidebar for file uploads
    st.sidebar.header("📁 Data Upload")
    st.sidebar.markdown("Upload your railway simulation CSV files:")
    
    before_file = st.sidebar.file_uploader(
        "Baseline Schedule (Before)",
        type=['csv'],
        help="Upload train_simulation_output_before.csv with original format"
    )
    
    after_file = st.sidebar.file_uploader(
        "Optimized Schedule (After)", 
        type=['csv'],
        help="Upload train_simulation_output_after.csv with original format"
    )
    
    if before_file is not None and after_file is not None:
        # Load data
        with st.spinner("Loading and processing data..."):
            df_before = analytics.load_data(before_file)
            df_after = analytics.load_data(after_file)
        
        if df_before is not None and df_after is not None:
            
            # Calculate metrics
            with st.spinner("Calculating comparative metrics..."):
                metrics_before = analytics.calculate_metrics(df_before)
                metrics_after = analytics.calculate_metrics(df_after)
            
            # Display KPI cards
            analytics.create_kpi_cards(metrics_before, metrics_after)
            
            st.markdown("---")
            
            # Create visualizations
            analytics.create_comparison_charts(df_before, df_after, metrics_before, metrics_after)
            
            st.markdown("---")
            
            # Track overlay visualization
            analytics.create_track_overlay(df_before, df_after)
            
            st.markdown("---")
            
            # Log audit trail and export reports
            st.subheader("📊 Reports & Analysis")
            
            improvements = analytics.log_audit_trail(
                metrics_before, metrics_after, 
                before_file.name, after_file.name
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                analytics.export_reports(metrics_before, metrics_after, improvements)
            
            with col2:
                if st.button("🔄 Refresh Analysis"):
                    st.rerun()
            
            # Generate insights
            st.markdown("---")
            st.subheader("💡 Key Insights")
            
            insights = analytics.generate_insights(improvements)
            
            st.markdown(f"""
            <div class="insight-box">
                {insights}
            </div>
            """, unsafe_allow_html=True)
            
            # Success message
            st.success("✅ Analysis completed successfully! Check the audit trail log for detailed records.")
    
    else:
        st.info("👆 Please upload both CSV files to begin the analysis")
        
        # Show sample data format
        st.subheader("📋 Expected Data Format")
        st.markdown("Your CSV files should contain the following columns:")
        
        sample_data = pd.DataFrame({
            'timestamp': ['1900-01-01 00:00:00', '1900-01-01 00:00:30'],
            'train_id': ['PA-074', 'SU-035'],
            'train_type': ['Passenger', 'Superfast'],
            'line': ['single_up', 'single_up'],
            'position_m': [466, 750],
            'speed_kmph': [56, 90],
            'station': ['', ''],
            'event': ['moving', 'moving'],
            'delay_minutes': [131, 188]
        })
        
        st.dataframe(sample_data)

if __name__ == "__main__":
    main()