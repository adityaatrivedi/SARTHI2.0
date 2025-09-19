#!/usr/bin/env python3
"""
Realistic Railway Simulation Data Generator
==========================================

Generates realistic baseline simulation data that represents:
- A busy but functioning railway system
- Realistic delays and operational challenges
- Believable headway violations and congestion
- Real-world train scheduling patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import List, Dict

class RealisticRailwaySimulator:
    def __init__(self):
        self.stations = {
            "Mumbai Central": {"position": 0, "platforms": 6},
            "Dadar": {"position": 15000, "platforms": 4}, 
            "Bandra": {"position": 25000, "platforms": 3},
            "Andheri": {"position": 35000, "platforms": 4},
            "Borivali": {"position": 50000, "platforms": 3},
            "Vasai": {"position": 70000, "platforms": 2},
            "Hoshangabad": {"position": 92000, "platforms": 3}
        }
        
        self.train_types = {
            "Express": {"speed": 120, "delay_prob": 0.15, "avg_delay": 8},
            "Superfast": {"speed": 140, "delay_prob": 0.10, "avg_delay": 5}, 
            "Passenger": {"speed": 80, "delay_prob": 0.25, "avg_delay": 12},
            "MEMU": {"speed": 90, "delay_prob": 0.20, "avg_delay": 10},
            "Mail": {"speed": 110, "delay_prob": 0.18, "avg_delay": 7},
            "Freight": {"speed": 60, "delay_prob": 0.35, "avg_delay": 15}
        }
        
        self.lines = ["single_up", "single_down", "central", "loop"]
        
    def generate_realistic_schedule(self, num_trains: int = 75, 
                                  simulation_duration_hours: int = 12) -> pd.DataFrame:
        """Generate realistic railway simulation data"""
        
        print(f"🚂 Generating realistic simulation data...")
        print(f"   Trains: {num_trains}")
        print(f"   Duration: {simulation_duration_hours} hours")
        
        # Create base time
        base_time = datetime(2024, 1, 15, 6, 0, 0)
        
        # Generate trains
        trains = []
        for i in range(num_trains):
            train_type = np.random.choice(list(self.train_types.keys()), 
                                        p=[0.2, 0.15, 0.3, 0.2, 0.1, 0.05])  # Realistic distribution
            
            train_id = f"{train_type[:2].upper()}-{i+1:03d}"
            trains.append({
                'train_id': train_id,
                'train_type': train_type,
                'config': self.train_types[train_type]
            })
        
        # Generate simulation records
        records = []
        
        # Create time snapshots every 30 seconds for realistic progression
        time_intervals = []
        for hour in range(simulation_duration_hours):
            for minute in [0, 15, 30, 45]:  # Every 15 minutes
                for second in [0, 30]:  # Every 30 seconds within each 15-min block
                    time_intervals.append(base_time + timedelta(hours=hour, minutes=minute, seconds=second))
        
        print(f"   Time intervals: {len(time_intervals)}")
        
        # Initialize train states
        train_states = {}
        for train in trains:
            # Random starting position and line
            initial_line = np.random.choice(self.lines)
            initial_position = np.random.uniform(0, 92000)
            
            # Determine if train starts with delay
            has_delay = np.random.random() < train['config']['delay_prob']
            initial_delay = np.random.exponential(train['config']['avg_delay']) if has_delay else 0
            
            train_states[train['train_id']] = {
                'train_type': train['train_type'],
                'line': initial_line,
                'position': initial_position,
                'speed': train['config']['speed'] + np.random.uniform(-10, 10),
                'delay_minutes': initial_delay,
                'event': 'moving' if np.random.random() > 0.1 else 'scheduled',
                'last_station': '',
                'direction': 1 if 'up' in initial_line or 'central' in initial_line else -1
            }
        
        # Generate records for each time interval
        for timestamp in time_intervals:
            self._update_train_states(train_states, timestamp)
            
            for train_id, state in train_states.items():
                # Determine station based on position
                station = self._get_station_from_position(state['position'])
                
                record = {
                    'timestamp': timestamp,
                    'train_id': train_id,
                    'train_type': state['train_type'],
                    'line': state['line'],
                    'position_m': round(state['position']),
                    'speed_kmph': round(state['speed'], 1),
                    'station': station if station else '',
                    'event': state['event'],
                    'delay_minutes': round(state['delay_minutes'], 1)
                }
                
                records.append(record)
        
        df = pd.DataFrame(records)
        
        # Add some realistic operational issues
        df = self._add_realistic_issues(df)
        
        print(f"✅ Generated {len(df)} realistic simulation records")
        return df
    
    def _update_train_states(self, train_states: Dict, timestamp: datetime):
        """Update train positions and states for given timestamp"""
        
        for train_id, state in train_states.items():
            # Update position based on speed and direction
            if state['event'] in ['moving']:
                # Convert speed from km/h to m/s, then to m per 30-second interval
                speed_ms = state['speed'] * 1000 / 3600  # m/s
                distance_increment = speed_ms * 30 * state['direction']  # 30 seconds
                
                state['position'] += distance_increment
                
                # Keep within track bounds
                state['position'] = max(0, min(92000, state['position']))
                
                # Reverse direction at ends
                if state['position'] <= 0 or state['position'] >= 92000:
                    state['direction'] *= -1
                    if state['line'] == 'single_up':
                        state['line'] = 'single_down'
                    elif state['line'] == 'single_down':
                        state['line'] = 'single_up'
            
            # Random events
            rand = np.random.random()
            
            if state['event'] == 'moving':
                # Small chance of delay or halt
                if rand < 0.02:  # 2% chance per interval
                    state['event'] = 'delayed'
                    state['delay_minutes'] += np.random.uniform(1, 5)
                elif rand < 0.005:  # 0.5% chance per interval
                    state['event'] = 'halted'
                    state['speed'] = 0
                elif rand < 0.01:  # 1% chance per interval
                    state['event'] = 'rerouted'
                    # Change line
                    available_lines = [l for l in self.lines if l != state['line']]
                    state['line'] = np.random.choice(available_lines)
            
            elif state['event'] == 'halted':
                # Chance to resume
                if rand < 0.3:  # 30% chance to resume per interval
                    state['event'] = 'moving'
                    state['speed'] = self.train_types[state['train_type']]['speed'] + np.random.uniform(-10, 10)
            
            elif state['event'] == 'delayed':
                # Usually resume quickly
                if rand < 0.5:  # 50% chance to resume per interval
                    state['event'] = 'moving'
            
            elif state['event'] == 'scheduled':
                # Start moving
                if rand < 0.4:  # 40% chance to start per interval
                    state['event'] = 'moving'
                    state['speed'] = self.train_types[state['train_type']]['speed'] + np.random.uniform(-10, 10)
            
            # Speed variations for moving trains
            if state['event'] == 'moving':
                state['speed'] += np.random.uniform(-2, 2)  # Small speed variations
                state['speed'] = max(20, min(160, state['speed']))  # Keep within reasonable bounds
    
    def _get_station_from_position(self, position: float) -> str:
        """Get station name based on position"""
        for station_name, station_info in self.stations.items():
            if abs(position - station_info['position']) < 1500:  # Within 1.5km
                return station_name
        return ""
    
    def _add_realistic_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add realistic operational issues to make the data more believable"""
        
        df = df.copy()
        
        # Add some congestion during peak hours
        peak_hours = [8, 9, 17, 18, 19]  # Morning and evening peaks
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        
        # Increase delays during peak hours
        peak_mask = df['hour'].isin(peak_hours)
        df.loc[peak_mask, 'delay_minutes'] *= 1.5
        
        # Add some freight train specific delays
        freight_mask = df['train_type'] == 'Freight'
        df.loc[freight_mask, 'delay_minutes'] *= 1.3
        
        # Ensure some minimum level of issues for realistic "before" scenario
        # Add controlled headway violations
        self._add_controlled_headway_violations(df)
        
        # Add some platform congestion
        self._add_platform_congestion(df)
        
        return df
    
    def _add_controlled_headway_violations(self, df: pd.DataFrame):
        """Add realistic but limited headway violations"""
        
        # Group by timestamp and line to find potential violations
        for (timestamp, line), group in df.groupby(['timestamp', 'line']):
            if len(group) > 1:
                # Sort by position
                sorted_group = group.sort_values('position_m')
                
                # For some groups, make trains closer than ideal (but not extreme)
                if np.random.random() < 0.15:  # 15% of groups have some congestion
                    positions = sorted_group['position_m'].values
                    
                    # Compress spacing to create realistic violations
                    compressed_positions = []
                    base_pos = positions[0]
                    
                    for i, pos in enumerate(positions):
                        if i == 0:
                            compressed_positions.append(pos)
                        else:
                            # Reduce spacing to 200-400m instead of ideal 500m+
                            spacing = np.random.uniform(200, 450)
                            new_pos = compressed_positions[-1] + spacing
                            compressed_positions.append(new_pos)
                    
                    # Update positions in dataframe
                    for idx, new_pos in zip(sorted_group.index, compressed_positions):
                        df.at[idx, 'position_m'] = new_pos
    
    def _add_platform_congestion(self, df: pd.DataFrame):
        """Add realistic platform congestion at major stations"""
        
        major_stations = ["Mumbai Central", "Dadar", "Bandra"]
        
        for station in major_stations:
            station_mask = df['station'] == station
            if station_mask.sum() > 0:
                # For some time periods, add congestion
                station_data = df[station_mask]
                
                for timestamp in station_data['timestamp'].unique():
                    time_mask = (df['station'] == station) & (df['timestamp'] == timestamp)
                    trains_at_station = df[time_mask]
                    
                    # If more trains than platforms, add delays
                    platform_capacity = self.stations[station]['platforms']
                    if len(trains_at_station) > platform_capacity:
                        excess_trains = len(trains_at_station) - platform_capacity
                        
                        # Add delays to excess trains
                        excess_indices = trains_at_station.index[-excess_trains:]
                        df.loc[excess_indices, 'delay_minutes'] += np.random.uniform(3, 8, excess_trains)
                        df.loc[excess_indices, 'event'] = 'delayed'

def main():
    """Generate realistic railway simulation data"""
    
    print("🚂 Realistic Railway Data Generator")
    print("=" * 50)
    
    simulator = RealisticRailwaySimulator()
    
    # Generate realistic baseline data
    df_realistic = simulator.generate_realistic_schedule(
        num_trains=75,
        simulation_duration_hours=12
    )
    
    # Save as new baseline
    output_file = "train_simulation_output_before.csv"
    df_realistic.to_csv(output_file, index=False)
    
    print(f"\n✅ Saved realistic simulation data to {output_file}")
    
    # Display statistics
    print(f"\n📊 DATA STATISTICS:")
    print(f"   Total records: {len(df_realistic):,}")
    print(f"   Unique trains: {df_realistic['train_id'].nunique()}")
    print(f"   Time span: {df_realistic['timestamp'].min()} to {df_realistic['timestamp'].max()}")
    
    # Event distribution
    event_dist = df_realistic['event'].value_counts()
    print(f"\n🚦 EVENT DISTRIBUTION:")
    for event, count in event_dist.items():
        percentage = (count / len(df_realistic)) * 100
        print(f"   {event}: {count:,} ({percentage:.1f}%)")
    
    # Train type distribution  
    train_dist = df_realistic['train_type'].value_counts()
    print(f"\n🚂 TRAIN TYPE DISTRIBUTION:")
    for train_type, count in train_dist.items():
        percentage = (count / len(df_realistic)) * 100
        print(f"   {train_type}: {count:,} ({percentage:.1f}%)")
    
    # Delay statistics
    delays = df_realistic[df_realistic['delay_minutes'] > 0]['delay_minutes']
    if len(delays) > 0:
        print(f"\n⏱️  DELAY STATISTICS:")
        print(f"   Trains with delays: {len(delays):,}")
        print(f"   Average delay: {delays.mean():.1f} minutes")
        print(f"   Max delay: {delays.max():.1f} minutes")
        print(f"   Delayed trains >5min: {len(delays[delays > 5]):,}")
    
    # Line usage
    line_dist = df_realistic['line'].value_counts()
    print(f"\n🛤️  LINE USAGE:")
    for line, count in line_dist.items():
        percentage = (count / len(df_realistic)) * 100
        print(f"   {line}: {count:,} ({percentage:.1f}%)")
    
    print(f"\n🎯 Ready for optimization!")
    print(f"   Run: python3 advanced_optimizer.py")
    print(f"   Then: streamlit run railway_dashboard.py")

if __name__ == "__main__":
    main()