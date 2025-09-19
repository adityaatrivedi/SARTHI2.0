import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def transform_simulation_data(input_file, output_file):
    """
    Transform the existing simulation data format to the required dashboard format
    """
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create required columns
    transformed_data = []
    
    # Get unique trains and stations
    trains = df['train_id'].unique()
    
    # Create station mapping based on position ranges
    def get_station_from_position(position):
        if position < 10000:
            return "Mumbai Central"
        elif position < 20000:
            return "Dadar"  
        elif position < 30000:
            return "Bandra"
        elif position < 50000:
            return "Andheri"
        elif position < 70000:
            return "Borivali"
        elif position < 90000:
            return "Vasai"
        else:
            return "Hoshangabad"
    
    # Process each train
    for train_id in trains:
        train_data = df[df['train_id'] == train_id].sort_values('timestamp')
        
        # Get train type
        train_type = train_data['train_type'].iloc[0]
        
        # Process each record for the train
        for idx, row in train_data.iterrows():
            # Generate station based on position
            station = get_station_from_position(row['position_m'])
            
            # Generate scheduled times (assume current time is actual + delay)
            timestamp = row['timestamp']
            delay_minutes = row['delay_minutes'] if pd.notna(row['delay_minutes']) else 0
            
            # Create scheduled arrival (subtract delay from actual)
            scheduled_arrival = timestamp - timedelta(minutes=delay_minutes)
            actual_arrival = timestamp
            
            # Create departure times (5-10 minutes after arrival)
            departure_buffer = np.random.randint(5, 11)
            scheduled_departure = scheduled_arrival + timedelta(minutes=departure_buffer)
            actual_departure = actual_arrival + timedelta(minutes=departure_buffer)
            
            # Determine platform and track based on train type and line
            if train_type in ['Express', 'Superfast', 'Mail']:
                platform_used = np.random.randint(1, 5)  # Main platforms
                track_used = 'Main'
            elif train_type == 'MEMU':
                platform_used = np.random.randint(5, 8)  # Side platforms
                track_used = 'Loop'
            elif train_type == 'Passenger':
                platform_used = np.random.randint(8, 12)  # Passenger platforms
                track_used = 'Local'
            else:  # Freight
                platform_used = np.random.randint(12, 15)  # Goods platforms
                track_used = 'Goods'
            
            # Override track based on line info if available
            if 'line' in df.columns and pd.notna(row['line']):
                if 'down' in str(row['line']).lower():
                    track_used = 'Main'
                elif 'up' in str(row['line']).lower():
                    track_used = 'Loop' 
                elif 'central' in str(row['line']).lower():
                    track_used = 'Central'
            
            transformed_data.append({
                'train_id': train_id,
                'train_type': train_type,
                'station': station,
                'scheduled_arrival': scheduled_arrival.strftime('%Y-%m-%d %H:%M:%S'),
                'actual_arrival': actual_arrival.strftime('%Y-%m-%d %H:%M:%S'),
                'scheduled_departure': scheduled_departure.strftime('%Y-%m-%d %H:%M:%S'),
                'actual_departure': actual_departure.strftime('%Y-%m-%d %H:%M:%S'),
                'delay_minutes': delay_minutes,
                'platform_used': platform_used,
                'track_used': track_used,
                'position_meters': row['position_m'],
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S')
            })
    
    # Create DataFrame and save
    transformed_df = pd.DataFrame(transformed_data)
    transformed_df.to_csv(output_file, index=False)
    
    print(f"Transformed data saved to {output_file}")
    print(f"Records: {len(transformed_df)}")
    print(f"Unique trains: {transformed_df['train_id'].nunique()}")
    print(f"Stations: {sorted(transformed_df['station'].unique())}")
    
    return transformed_df

if __name__ == "__main__":
    # Transform both files
    print("Transforming simulation data files...")
    
    transform_simulation_data(
        'train_simulation_output_before.csv',
        'train_simulation_output_before_transformed.csv'
    )
    
    transform_simulation_data(
        'train_simulation_output_after.csv', 
        'train_simulation_output_after_transformed.csv'
    )
    
    print("Data transformation completed!")