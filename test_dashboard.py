#!/usr/bin/env python3

import pandas as pd
import sys
import os

def test_dashboard():
    """Test the dashboard with original data format"""
    
    print("🧪 Testing Railway Dashboard with Original Data Format")
    print("=" * 55)
    
    # Check if data files exist
    before_file = "train_simulation_output_before.csv"
    after_file = "train_simulation_output_after.csv"
    
    if not os.path.exists(before_file):
        print(f"❌ Missing file: {before_file}")
        return False
        
    if not os.path.exists(after_file):
        print(f"❌ Missing file: {after_file}")
        return False
    
    print(f"✅ Found data files: {before_file}, {after_file}")
    
    # Load and validate data structure
    try:
        df_before = pd.read_csv(before_file)
        df_after = pd.read_csv(after_file)
        
        expected_columns = [
            'timestamp', 'train_id', 'train_type', 'line', 
            'position_m', 'speed_kmph', 'station', 'event', 'delay_minutes'
        ]
        
        # Check columns
        missing_before = set(expected_columns) - set(df_before.columns)
        missing_after = set(expected_columns) - set(df_after.columns)
        
        if missing_before:
            print(f"❌ Missing columns in {before_file}: {missing_before}")
            return False
            
        if missing_after:
            print(f"❌ Missing columns in {after_file}: {missing_after}")
            return False
            
        print("✅ All required columns present")
        
        # Basic data validation
        print(f"📊 Before data: {len(df_before)} records, {df_before['train_id'].nunique()} unique trains")
        print(f"📊 After data: {len(df_after)} records, {df_after['train_id'].nunique()} unique trains")
        
        # Check event types
        events_before = df_before['event'].value_counts()
        events_after = df_after['event'].value_counts()
        
        print(f"📈 Event types (Before): {list(events_before.index)}")
        print(f"📈 Event types (After): {list(events_after.index)}")
        
        # Check train types
        train_types = df_before['train_type'].unique()
        print(f"🚂 Train types: {list(train_types)}")
        
        # Test metrics calculation
        from railway_dashboard import RailwayAnalytics
        
        analytics = RailwayAnalytics()
        
        # Process data
        df_before['timestamp'] = pd.to_datetime(df_before['timestamp'])
        df_before['delay_minutes'] = df_before['delay_minutes'].fillna(0)
        df_before['station'] = df_before['station'].fillna('Unknown')
        
        df_after['timestamp'] = pd.to_datetime(df_after['timestamp'])
        df_after['delay_minutes'] = df_after['delay_minutes'].fillna(0)
        df_after['station'] = df_after['station'].fillna('Unknown')
        
        # Calculate metrics
        metrics_before = analytics.calculate_metrics(df_before)
        metrics_after = analytics.calculate_metrics(df_after)
        
        print("\n📊 Calculated Metrics:")
        print("-" * 40)
        for key, value in metrics_before.items():
            after_value = metrics_after[key]
            change = after_value - value
            print(f"{key}: {value:.1f} → {after_value:.1f} ({change:+.1f})")
        
        print("\n✅ Dashboard test completed successfully!")
        print("🚀 Ready to run: streamlit run railway_dashboard.py")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_dashboard()
    sys.exit(0 if success else 1)