
import csv
import random
import datetime
import math
import pandas as pd

# --- 1. SIMULATION CONFIGURATION ---

# Core Parameters
SIMULATION_DURATION_HOURS = 8
TIME_STEP_SECONDS = 30
HEADWAY_METERS = 500
ROUTE_LENGTH_M = 92000

# Realism Constraints (Target Percentages)
MIN_DELAYED_TRAIN_PERCENT = 0.25
MIN_HALTED_TRAIN_PERCENT = 0.10
MIN_REROUTED_TRAIN_PERCENT = 0.05

# Train Categories: { type: (min_speed_kmph, max_speed_kmph, dwell_time_minutes, priority) }
# Priority: Lower number is higher priority
TRAIN_CATEGORIES = {
    "Superfast": (90, 130, 2, 1),
    "Express": (80, 110, 5, 2),
    "Mail": (70, 100, 5, 3),
    "MEMU": (60, 90, 3, 4),
    "Passenger": (50, 80, 7, 5),
    "Freight": (40, 70, 30, 6),
}

# Event Probabilities
PROB_TECHNICAL_DELAY = 0.0001 # Per train per time step
PROB_MAINTENANCE_BLOCK = 0.00001 # Per time step
DISRUPTION_PROBABILITIES = {
    "Superfast": 0.00005,
    "Express": 0.0001,
    "Mail": 0.0001,
    "MEMU": 0.00015,
    "Passenger": 0.0002,
    "Freight": 0.0003,
}


# --- 2. DATA LOADING ---

def load_csv_data(filepath):
    with open(filepath, mode='r', encoding='utf-8') as file:
        return list(csv.DictReader(file))

STATIONS_DATA = load_csv_data('stations.csv')
TRACKS_DATA = load_csv_data('tracks.csv')


# --- 3. CORE SIMULATION CLASSES ---

class Station:
    def __init__(self, data):
        self.id = data['station_id']
        self.name = data['name']
        self.position_m = int(data['position_m'])
        self.platforms = {i: None for i in range(1, int(data['platforms']) + 1)} # platform_id: train_id

    def get_free_platform(self):
        for p_id, t_id in self.platforms.items():
            if t_id is None:
                return p_id
        return None

    def occupy_platform(self, platform_id, train_id):
        if self.platforms.get(platform_id) is None:
            self.platforms[platform_id] = train_id
            return True
        return False

    def release_platform(self, train_id):
        for p_id, t_id in self.platforms.items():
            if t_id == train_id:
                self.platforms[p_id] = None
                return

class Train:
    def __init__(self, train_id, train_type, direction, scheduled_departure_time):
        self.id = train_id
        self.type = train_type
        self.direction = direction # "up" (Bhopal -> Itarsi) or "down" (Itarsi -> Bhopal)
        self.min_speed, self.max_speed, self.dwell_time, self.priority = TRAIN_CATEGORIES[train_type]
        self.disruption_probability = DISRUPTION_PROBABILITIES[train_type]

        # Dynamic State
        self.line = "single_down" if self.direction == "down" else "single_up"
        self.position_m = ROUTE_LENGTH_M if self.direction == "down" else 0
        self.speed_kmph = 0
        self.event = "scheduled"
        self.station = None # Name of station if docked
        self.delay_minutes = 0
        
        # Flags & Timers
        self.is_rerouted = False
        self.has_been_halted = False
        self.has_been_delayed = False
        self.is_finished = False
        self.dwell_timer = 0
        self.halt_timer = 0

        # Scheduling
        self.scheduled_departure_time = scheduled_departure_time
        self.actual_departure_time = None
        self.planned_platform = None

    def log_state(self, timestamp):
        return {
            "timestamp": timestamp,
            "train_id": self.id,
            "train_type": self.type,
            "line": self.line,
            "position_m": int(self.position_m),
            "speed_kmph": int(self.speed_kmph),
            "station": self.station,
            "event": self.event,
            "delay_minutes": int(self.delay_minutes),
        }

class Simulation:
    def __init__(self, num_trains=80, trains=None):
        self.stations = {s['name']: Station(s) for s in STATIONS_DATA}
        self.tracks = {t['line_id']: t for t in TRACKS_DATA}
        if trains:
            self.trains = trains
        else:
            self.trains = self._generate_trains(num_trains)
        self.simulation_log = []
        self.maintenance_blocks = {} # line_id: end_time

        # For realism stats
        self.halted_trains = set()
        self.rerouted_trains = set()
        self.delayed_trains = set()

    def _generate_trains(self, num_trains):
        trains = []
        start_time = datetime.datetime.strptime("00:00:00", '%H:%M:%S')
        for i in range(num_trains):
            train_type = random.choice(list(TRAIN_CATEGORIES.keys()))
            direction = random.choice(["up", "down"])
            # Stagger departures over the simulation duration
            scheduled_departure = start_time + datetime.timedelta(seconds=random.randint(0, SIMULATION_DURATION_HOURS * 3600 - 3600))
            train_id = f"{train_type[:2].upper()}-{i+1:03d}"
            trains.append(Train(train_id, train_type, direction, scheduled_departure))
        return sorted(trains, key=lambda t: t.scheduled_departure_time)

    def _get_conflicting_trains(self, current_train):
        conflicts = []
        for train in self.trains:
            if train.id == current_train.id or train.is_finished or train.line != current_train.line:
                continue
            
            distance = abs(current_train.position_m - train.position_m)
            if distance >= HEADWAY_METERS:
                continue

            # Logic for unidirectional lines (single_up, single_down, loop)
            if self.tracks[current_train.line]['directionality'] == 'unidirectional':
                is_in_front = (train.position_m > current_train.position_m) if current_train.direction == "up" else (train.position_m < current_train.position_m)
                if is_in_front:
                    conflicts.append(train)
            
            # Logic for bidirectional line (central)
            else: # bidirectional
                is_moving_towards = (current_train.direction == "up" and train.direction == "down" and current_train.position_m < train.position_m) or \
                                    (current_train.direction == "down" and train.direction == "up" and current_train.position_m > train.position_m)
                
                is_same_direction_in_front = (current_train.direction == train.direction) and \
                                             ((current_train.direction == "up" and train.position_m > current_train.position_m) or \
                                              (current_train.direction == "down" and train.position_m < current_train.position_m))

                if is_moving_towards or is_same_direction_in_front:
                    conflicts.append(train)
        
        return conflicts

    def _find_station_at(self, position_m):
        for station in self.stations.values():
            if abs(position_m - station.position_m) < 200: # 200m tolerance for arrival
                return station
        return None

    def _should_reroute(self, train):
        if train.is_rerouted or train.line not in ['single_up', 'single_down']:
            return False

        # Check if central line is free
        for other_train in self.trains:
            if other_train.id != train.id and not other_train.is_finished and other_train.line == 'central':
                distance = abs(train.position_m - other_train.position_m)
                if distance < HEADWAY_METERS * 2: # Use a larger buffer for rerouting
                    return False
        return True

    def reallocate_platform(self, train_id, station_name, new_platform_id):
        train = next((t for t in self.trains if t.id == train_id), None)
        station = self.stations.get(station_name)
        if train and station and new_platform_id in station.platforms:
            train.planned_platform = new_platform_id
            return True
        return False

    def run(self):
        start_sim_time = datetime.datetime.strptime("00:00:00", '%H:%M:%S')
        sim_end_time = start_sim_time + datetime.timedelta(hours=SIMULATION_DURATION_HOURS)
        
        current_sim_time = start_sim_time
        while current_sim_time < sim_end_time:
            timestamp_str = current_sim_time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Handle maintenance blocks
            self._update_maintenance_blocks(current_sim_time)

            for train in self.trains:
                if train.is_finished:
                    continue
                self._update_train_state(train, current_sim_time)
                self.simulation_log.append(train.log_state(timestamp_str))

            current_sim_time += datetime.timedelta(seconds=TIME_STEP_SECONDS)

        return pd.DataFrame(self.simulation_log)

    def _update_train_state(self, train, current_time):
        # --- Event Handling ---
        
        # 1. Initial Departure
        if train.event == "scheduled" and current_time >= train.scheduled_departure_time:
            train.actual_departure_time = current_time
            train.delay_minutes = (current_time - train.scheduled_departure_time).total_seconds() / 60
            if train.delay_minutes > 5: self.delayed_trains.add(train.id)
            
            start_station_name = "Bhopal Jn" if train.direction == "up" else "Itarsi Jn"
            start_station = self.stations[start_station_name]
            
            p_id = start_station.get_free_platform()
            if p_id:
                start_station.occupy_platform(p_id, train.id) # Momentarily occupy for departure logic
                start_station.release_platform(train.id)
                train.event = "departed"
                train.station = None
                train.speed_kmph = random.randint(train.min_speed, train.max_speed)
            else:
                train.event = "delayed" # No platform free, delayed at origin
                train.delay_minutes += TIME_STEP_SECONDS / 60
                return

        if train.event == "scheduled": return

        # 2. Dwell time at station
        if train.event == "arrived":
            train.dwell_timer -= TIME_STEP_SECONDS
            if train.dwell_timer <= 0:
                station = self.stations[train.station]
                station.release_platform(train.id)
                train.station = None
                train.event = "departed"
                train.speed_kmph = random.randint(train.min_speed, train.max_speed)
            else:
                return # Still dwelling

        # 3. Halt timer for delays/halts
        if train.halt_timer > 0:
            train.halt_timer -= TIME_STEP_SECONDS
            train.delay_minutes += TIME_STEP_SECONDS / 60
            if train.delay_minutes > 5: self.delayed_trains.add(train.id)
            return # Still halted

        # --- Conflict & Movement ---

        # 4. Check for line blockages
        if train.line in self.maintenance_blocks:
            train.event = "halted"
            train.speed_kmph = 0
            train.halt_timer = TIME_STEP_SECONDS # Wait for next check
            self.halted_trains.add(train.id)
            train.has_been_halted = True
            return

        # 5. Headway and Conflict Resolution
        conflicts = self._get_conflicting_trains(train)
        if conflicts:
            # Higher priority train proceeds, lower priority one halts
            highest_priority_conflict = min(conflicts, key=lambda c: c.priority)
            if train.priority > highest_priority_conflict.priority:
                train.event = "halted"
                train.speed_kmph = 0
                train.halt_timer = TIME_STEP_SECONDS
                self.halted_trains.add(train.id)
                train.has_been_halted = True
                
                # Rerouting logic
                if self._should_reroute(train):
                    train.line = 'central'
                    train.event = 'rerouted'
                    self.rerouted_trains.add(train.id)
                    train.is_rerouted = True
                return

        # 6. If no conflicts, resume movement
        if train.event == "halted":
            train.event = "moving"
            train.speed_kmph = random.randint(train.min_speed, train.max_speed)

        # 7. Update Position
        if train.speed_kmph > 0:
            train.event = "moving"
            distance_change_m = (train.speed_kmph * 1000 / 3600) * TIME_STEP_SECONDS
            train.position_m += distance_change_m if train.direction == "up" else -distance_change_m

        # 8. Station Arrival
        station = self._find_station_at(train.position_m)
        if station and train.station is None and train.event == "moving":
            platform_to_occupy = None
            if train.planned_platform and station.platforms.get(train.planned_platform) is None:
                platform_to_occupy = train.planned_platform
            else:
                platform_to_occupy = station.get_free_platform()

            if platform_to_occupy:
                train.position_m = station.position_m
                train.speed_kmph = 0
                train.event = "arrived"
                train.station = station.name
                station.occupy_platform(platform_to_occupy, train.id)
                train.dwell_timer = train.dwell_time * 60
            else: # No platform, halt before station
                train.event = "halted"
                train.speed_kmph = 0
                train.halt_timer = TIME_STEP_SECONDS
                self.halted_trains.add(train.id)
                train.has_been_halted = True

        # 9. Random Technical Delays
        if train.event == "moving" and random.random() < train.disruption_probability:
            delay_duration = random.randint(5, 15) * 60
            train.halt_timer = delay_duration
            train.event = "delayed"
            train.speed_kmph = 0
            train.delay_minutes += delay_duration / 60
            if train.delay_minutes > 5: self.delayed_trains.add(train.id)
            train.has_been_delayed = True

        # 10. Finish route
        if (train.direction == "up" and train.position_m >= ROUTE_LENGTH_M) or \
           (train.direction == "down" and train.position_m <= 0):
            train.is_finished = True
            train.event = "finished"
            train.speed_kmph = 0
            final_station_name = "Itarsi Jn" if train.direction == "up" else "Bhopal Jn"
            train.station = final_station_name
            train.position_m = self.stations[final_station_name].position_m


    def _update_maintenance_blocks(self, current_time):
        # Clear expired blocks
        for line, end_time in list(self.maintenance_blocks.items()):
            if current_time >= end_time:
                del self.maintenance_blocks[line]
        
        # Add new blocks
        if random.random() < PROB_MAINTENANCE_BLOCK:
            line_to_block = random.choice(['single_up', 'single_down', 'central'])
            if line_to_block not in self.maintenance_blocks:
                block_duration = datetime.timedelta(minutes=random.randint(15, 60))
                self.maintenance_blocks[line_to_block] = current_time + block_duration
