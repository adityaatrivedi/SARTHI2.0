import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import matplotlib.dates as mdates
import plotly.graph_objects as go
from matplotlib.lines import Line2D

# --- Configuration ---
OUTPUT_DIR = 'outputs'
STATIONS_FILE = 'stations.csv'
SIMULATION_FILE = 'train_simulation_output.csv'
ROUTE_LENGTH_M = 92000


def load_data():
    """Loads and prepares the simulation and station data."""
    if not os.path.exists(SIMULATION_FILE):
        print(f"Error: '{SIMULATION_FILE}' not found. Please run the simulation first.")
        return None, None

    df = pd.read_csv(SIMULATION_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['position_km'] = df['position_m'] / 1000

    stations_df = pd.read_csv(STATIONS_FILE)
    stations_df['position_km'] = stations_df['position_m'] / 1000
    
    return df, stations_df

def plot_train_movement_timeline(df, stations_df):
    """Generates a plot showing the position of each train over time."""
    plt.figure(figsize=(18, 10))
    ax = plt.gca() # Get current axes
    sns.lineplot(data=df, x='timestamp', y='position_km', hue='train_id', legend=None, ax=ax)

    # Add station lines and labels
    for _, station in stations_df.iterrows():
        ax.axhline(y=station['position_km'], color='r', linestyle='--', linewidth=0.8)
        ax.text(df['timestamp'].min(), station['position_km'], f" {station['name']}", va='center', color='red', alpha=0.9)

    plt.title('Train Movement Timeline (Bhopal to Itarsi)')
    plt.xlabel('Time')
    plt.ylabel('Position (km)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'train_movement_timeline.png'))
    plt.close()
    print("Generated: Train Movement Timeline")

def plot_delay_distribution(df):
    """Generates a histogram of final train delays."""
    final_delays = df.loc[df.groupby('train_id')['timestamp'].idxmax()]
    plt.figure(figsize=(10, 6))
    sns.histplot(final_delays['delay_minutes'], bins=20, kde=True)
    plt.title('Distribution of Final Train Delays')
    plt.xlabel('Delay (minutes)')
    plt.ylabel('Number of Trains')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'delay_distribution.png'))
    plt.close()
    print("Generated: Delay Distribution Plot")

def plot_average_delay_by_type(df):
    """Generates a bar chart of average final delay by train type."""
    final_delays = df.loc[df.groupby('train_id')['timestamp'].idxmax()]
    avg_delay = final_delays.groupby('train_type')['delay_minutes'].mean().sort_values()

    plt.figure(figsize=(12, 7))
    avg_delay.plot(kind='bar', color=sns.color_palette('viridis', len(avg_delay)))
    plt.title('Average Final Delay by Train Type')
    plt.xlabel('Train Type')
    plt.ylabel('Average Delay (minutes)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'average_delay.png'))
    plt.close()
    print("Generated: Average Delay by Train Type Plot")

def plot_line_utilization(df):
    """Generates a bar chart showing the usage of each track line."""
    # Count time steps spent on each line
    line_usage = df['line'].value_counts()
    plt.figure(figsize=(10, 6))
    line_usage.plot(kind='bar', color=sns.color_palette('muted'))
    plt.title('Track Line Utilization (Time Steps Spent on Each Line)')
    plt.xlabel('Line Type')
    plt.ylabel('Number of Time Steps Recorded')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'line_utilization.png'))
    plt.close()
    print("Generated: Line Utilization Plot")

def plot_event_frequency(df):
    """Generates a bar chart of event frequencies."""
    event_counts = df['event'].value_counts()
    plt.figure(figsize=(12, 7))
    event_counts.plot(kind='bar', color=sns.color_palette('plasma'))
    plt.title('Frequency of Simulation Events')
    plt.xlabel('Event Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'event_frequency.png'))
    plt.close()
    print("Generated: Event Frequency Plot")

def plot_station_vs_time_graph(df, stations_df):
    """
    Generates a position-vs-time graph for 10 evenly-spaced trains, with a dual
    Y-axis for stations and line style changes for halts.
    """
    print("Generating: Final Graph with Dotted Halts and Spaced-Out Trains")
    
    # --- 1. Spaced-Out, Deterministic Train Selection ---
    all_train_ids = sorted(df['train_id'].unique())
    if len(all_train_ids) > 10:
        indices = np.linspace(0, len(all_train_ids) - 1, 10, dtype=int)
        trains_to_plot = [all_train_ids[i] for i in indices]
    else:
        trains_to_plot = all_train_ids
    sample_df = df[df['train_id'].isin(trains_to_plot)]

    # --- 2. Plotting Setup ---
    fig, ax = plt.subplots(figsize=(20, 12))
    
    colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(trains_to_plot)))
    train_color_map = dict(zip(trains_to_plot, colors))
    legend_elements = []

    train_types = sample_df.drop_duplicates(subset=['train_id']).set_index('train_id')['train_type'].to_dict()

    # --- 3. Plot Each Train's Path with Style Changes ---
    for train_id in trains_to_plot:
        train_data = sample_df[sample_df['train_id'] == train_id].sort_values('timestamp')
        color = train_color_map[train_id]
        
        # Plot segment by segment to change linestyle for halts
        for i in range(len(train_data) - 1):
            p1 = train_data.iloc[i]
            p2 = train_data.iloc[i+1]
            
            # A halt is when position doesn't change
            is_halted = (p1['position_km'] == p2['position_km'])
            linestyle = ':' if is_halted else '-'
            
            ax.plot([p1['timestamp'], p2['timestamp']], [p1['position_km'], p2['position_km']], 
                    color=color, linestyle=linestyle, linewidth=2)

        label = f"{train_types.get(train_id, '')} ({train_id})"
        legend_elements.append(Line2D([0], [0], color=color, lw=2, label=label))

    # --- 4. Formatting and Dual Y-Axis ---
    ax.set_title('Train Movement and Halts')
    ax.set_xlabel('Time')
    ax.set_ylabel('Position (km) from Bhopal', fontsize=12)

    # X-axis time formatting
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Primary Y-axis (Position)
    ax.set_ylim(bottom=-2, top=ROUTE_LENGTH_M / 1000 + 2)
    ax.grid(which='major', axis='x', linestyle='-', alpha=0.6)

    # Secondary Y-axis for station names
    ax2 = ax.twinx()
    station_positions = stations_df['position_km'].tolist()
    station_names = stations_df['name'].tolist()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(station_positions)
    ax2.set_yticklabels(station_names)
    ax2.set_ylabel('Stations', fontsize=12)

    # Add horizontal grid lines for stations on the primary axis
    for pos in station_positions:
        ax.axhline(y=pos, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)

    # --- 5. Legend ---
    # Add custom lines for legend to explain linestyles
    legend_elements.insert(0, Line2D([0], [0], color='gray', lw=2, linestyle='-', label='Moving'))
    legend_elements.insert(1, Line2D([0], [0], color='gray', lw=2, linestyle=':', label='Halted'))
    ax.legend(handles=legend_elements, title="Trains & Status", bbox_to_anchor=(1.02, 1), loc='upper left')

    fig.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend
    plt.savefig(os.path.join(OUTPUT_DIR, 'station_vs_time.png'))
    plt.close()
    print("Generated: Final Graph with Dotted Halts and Spaced-Out Trains")

def generate_train_graph(df):
    """
    Generates a Time-Distance Chart for train movements using Plotly.

    This function creates a Plotly figure that visualizes train paths,
    showing both movement between stations and dwell times at stations.

    Args:
        df (pd.DataFrame):
            A DataFrame containing the train schedule data. The DataFrame must
            include the following columns:
            - 'train_id': Unique identifier for each train.
            - 'train_type': Category of the train (e.g., 'Express', 'Freight').
            - 'station': Name of the station.
            - 'arrival_time': The arrival time at the station (datetime object).
            - 'departure_time': The departure time from the station (datetime object).
            - 'delay_reason': A string describing the reason for any delay.

            For correct station ordering on the Y-axis, the 'station' column
            should be a pandas Categorical type with the desired order set, e.g.:
            station_order = ['Station A', 'Station B', 'Station C']
            df['station'] = pd.Categorical(df['station'], categories=station_order, ordered=True)

    Returns:
        go.Figure:
            A Plotly figure object visualizing the train graph. This figure can be
            displayed using fig.show() or integrated into a Dash application.
    """
    fig = go.Figure()

    # Create a color map for different train types for consistent coloring
    unique_train_types = df['train_type'].unique()
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3']
    color_map = {train_type: colors[i % len(colors)] for i, train_type in enumerate(unique_train_types)}

    # Iterate through each unique train to create its path as a separate trace
    for train_id in df['train_id'].unique():
        train_data = df[df['train_id'] == train_id].sort_values('arrival_time')

        if train_data.empty:
            continue

        # --- Core Logic: Create stop-and-go data structure ---
        x_data = []
        y_data = []
        hover_texts = []

        for _, row in train_data.iterrows():
            x_data.extend([row['arrival_time'], row['departure_time']])
            y_data.extend([row['station'], row['station']])

            base_hover_info = (
                f"<b>Train ID:</b> {row['train_id']}<br>"
                f"<b>Type:</b> {row['train_type']}<br>"
                f"<b>Delay Reason:</b> {row['delay_reason']}"
            )
            hover_texts.append(f"{base_hover_info}<br><b>Arrival:</b> {row['arrival_time'].strftime('%H:%M')}")
            hover_texts.append(f"{base_hover_info}<br><b>Departure:</b> {row['departure_time'].strftime('%H:%M')}")

        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='lines',
            name=train_data['train_type'].iloc[0],
            line=dict(color=color_map.get(train_data['train_type'].iloc[0])),
            hoverinfo='text',
            hovertext=hover_texts,
            legendgroup=train_data['train_type'].iloc[0],
            showlegend=not any(trace.name == train_data['train_type'].iloc[0] for trace in fig.data)
        ))

    # --- Layout and Axes Configuration ---
    fig.update_layout(
        title_text="Train Movement Graph (Time-Distance Chart)",
        xaxis_title="Time",
        yaxis_title="Station",
        yaxis=dict(
            type='category'
        ),
        legend_title="Train Type",
        template="plotly_white"
    )

    return fig

def main():
    """Main function to run the analysis."""
    print("Starting analysis of simulation data...")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    df, stations_df = load_data()

    if df is not None:
        # --- Matplotlib Plots ---
        plot_train_movement_timeline(df, stations_df)
        plot_station_vs_time_graph(df, stations_df)
        plot_delay_distribution(df)
        plot_average_delay_by_type(df)
        plot_line_utilization(df)
        plot_event_frequency(df)
        print("\nMatplotlib plots saved in 'outputs/' directory.")

        # --- Interactive Plotly Plot ---
        print("\nGenerating interactive Plotly graph...")
        
        # 1. Prepare data for the new function
        # Select the first 5 trains for clarity
        trains_to_plot = df['train_id'].unique()[:5]
        sample_df = df[df['train_id'].isin(trains_to_plot)]

        station_order = stations_df.sort_values('position_m')['name'].tolist()
        
        # Use the sampled data for aggregation
        station_stops_df = sample_df[sample_df['station'].notna()].groupby(['train_id', 'train_type', 'station']).agg(
            arrival_time=('timestamp', 'min'),
            departure_time=('timestamp', 'max')
        ).reset_index()

        delayed_trains = df[df['event'].isin(['delayed', 'halted'])]['train_id'].unique()
        station_stops_df['delay_reason'] = station_stops_df['train_id'].apply(
            lambda x: 'Delayed/Halted' if x in delayed_trains else 'On Time'
        )

        station_stops_df['station'] = pd.Categorical(
            station_stops_df['station'],
            categories=station_order,
            ordered=True
        )
        station_stops_df = station_stops_df.sort_values(by=['train_id', 'arrival_time'])

        # 2. Generate and save the Plotly figure
        if not station_stops_df.empty:
            plotly_fig = generate_train_graph(station_stops_df)
            output_path = os.path.join(OUTPUT_DIR, 'train_movement_plotly.html')
            plotly_fig.write_html(output_path)
            print(f"Generated interactive Plotly graph: {output_path}")


if __name__ == '__main__':
    main()